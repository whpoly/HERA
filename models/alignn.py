"""ALIGNN variants, including heterogeneous and region-hypergraph hybrids."""

from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from .modules import (
    AtomTypeGlobalAttentionReadout,
    DefectAwareGateConv,
    RelationFusionUpdate,
)
from .hypergraph import RegionHypergraphInteraction
from ..config.defaults import validate_alignn_hetero_relations


def _edge_type_key(edge_type):
    return "__".join(edge_type)


def _edge_parameter_key(edge_type):
    """Return a distinct parameter key for each directed edge type."""
    return _edge_type_key(edge_type)


def _empty_index(reference):
    return torch.empty((2, 0), dtype=torch.long, device=reference.device)


def _normalize_aggr(aggr):
    if aggr == "sum":
        return "add"
    if aggr in {"add", "mean", "max", "min"}:
        return aggr
    return "add"


def _pool_mean_or_zeros(features, batch, dim_size, width, reference):
    if features is None or batch is None or features.size(0) == 0:
        return reference.new_zeros((dim_size, width))

    out = reference.new_zeros((dim_size, features.size(-1)))
    counts = reference.new_zeros((dim_size, 1))
    out.index_add_(0, batch.long(), features.to(dtype=out.dtype))
    counts.index_add_(
        0,
        batch.long(),
        torch.ones(
            (features.size(0), 1),
            device=features.device,
            dtype=counts.dtype,
        ),
    )
    return out / counts.clamp_min(1.0)


def _graph_count(batch_dict=None, batch=None, state=None):
    if state is not None:
        return int(state.size(0))
    if batch is not None and batch.numel() > 0:
        return int(batch.max().item()) + 1
    if batch_dict is not None:
        counts = [int(value.max().item()) + 1 for value in batch_dict.values() if value.numel() > 0]
        if counts:
            return max(counts)
    return 1


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that falls back to running stats for single-row inputs."""

    def forward(self, input):
        values_per_channel = input.numel() // input.size(1) if input.dim() >= 2 else input.numel()
        if self.training and values_per_channel <= 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(input)


def _make_feature_norm(channels, normalization):
    normalization = str(normalization).lower()
    if normalization == "layernorm":
        return nn.LayerNorm(channels)
    if normalization == "batchnorm":
        return SafeBatchNorm1d(channels)
    if normalization == "none":
        return nn.Identity()
    raise ValueError("normalization must be one of: layernorm, batchnorm, none")


class MLPLayer(nn.Module):
    """ALIGNN helper: Linear, normalization, SiLU."""

    def __init__(self, in_features, out_features, normalization="batchnorm"):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            _make_feature_norm(out_features, normalization),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.layer(x)


class RBFExpansion(nn.Module):
    """Gaussian radial basis expansion used by the old official ALIGNN."""

    def __init__(self, vmin, vmax, bins, sigma=None):
        super().__init__()
        centers = torch.linspace(float(vmin), float(vmax), int(bins))
        self.register_buffer("centers", centers)
        if sigma is None:
            sigma = (float(vmax) - float(vmin)) / max(int(bins) - 1, 1)
        self.sigma = float(sigma)

    @property
    def out_features(self):
        return int(self.centers.numel())

    def forward(self, values):
        return torch.exp(-((values.view(-1, 1) - self.centers.view(1, -1)) / self.sigma) ** 2)


class AngleExpansion(RBFExpansion):
    """Official ALIGNN angle expansion over bond-angle cosine values."""

    def __init__(self, num_centers=40, sigma=None):
        super().__init__(-1.0, 1.0, num_centers, sigma=sigma)


def _official_feature_embedding(in_features, hidden_dim, normalization="batchnorm"):
    return nn.Sequential(
        MLPLayer(in_features, hidden_dim, normalization=normalization),
        MLPLayer(hidden_dim, hidden_dim, normalization=normalization),
    )


def _embed_distance_edges(edge_attr, edge_vec, distance_expansion, edge_embedding):
    if edge_vec is not None and edge_vec.size(0) == edge_attr.size(0):
        bond_length = edge_vec.float().norm(dim=-1)
        edge_attr = distance_expansion(bond_length)
    else:
        edge_attr = edge_attr.float()
    return edge_embedding(edge_attr)


def _angle_features_from_pairs(pair_src, pair_dst, edge_vec, angle_expansion):
    if len(pair_src) == 0:
        return _empty_index(edge_vec), edge_vec.new_empty((0, angle_expansion.out_features))

    src = torch.tensor(pair_src, dtype=torch.long, device=edge_vec.device)
    dst = torch.tensor(pair_dst, dtype=torch.long, device=edge_vec.device)
    if edge_vec.numel() == 0:
        return _empty_index(edge_vec), edge_vec.new_empty((0, angle_expansion.out_features))

    v_in = -edge_vec[src]
    v_out = edge_vec[dst]
    in_norm = v_in.norm(dim=-1)
    out_norm = v_out.norm(dim=-1)
    valid = (in_norm > 1e-8) & (out_norm > 1e-8)
    if valid.sum().item() == 0:
        return _empty_index(edge_vec), edge_vec.new_empty((0, angle_expansion.out_features))

    src = src[valid]
    dst = dst[valid]
    v_in = v_in[valid]
    v_out = v_out[valid]
    cos_angle = F.cosine_similarity(v_in, v_out, dim=-1).clamp(-1.0, 1.0)
    return torch.stack([src, dst], dim=0), angle_expansion(cos_angle)


def _valid_edge_mask(edge_vec):
    if edge_vec.numel() == 0:
        return []
    return (edge_vec.detach().norm(dim=-1) > 1e-8).cpu().tolist()


def build_line_graph(edge_index, edge_vec, angle_expansion):
    """Build directed bond-angle line graph edges for a homogeneous graph."""

    if edge_index.size(1) == 0:
        return _empty_index(edge_index), edge_index.new_empty((0, angle_expansion.out_features), dtype=torch.float)

    sources = edge_index[0].detach().cpu().tolist()
    targets = edge_index[1].detach().cpu().tolist()
    valid_edges = _valid_edge_mask(edge_vec)
    outgoing = defaultdict(list)
    for edge_id, src in enumerate(sources):
        if valid_edges[edge_id]:
            outgoing[int(src)].append(edge_id)

    pair_src = []
    pair_dst = []
    for first_edge, shared_node in enumerate(targets):
        if not valid_edges[first_edge]:
            continue
        for second_edge in outgoing.get(int(shared_node), []):
            if first_edge == second_edge:
                continue
            pair_src.append(first_edge)
            pair_dst.append(second_edge)

    return _angle_features_from_pairs(pair_src, pair_dst, edge_vec, angle_expansion)


def build_hetero_line_graph(
        edge_index_dict,
        edge_vec_all,
        edge_offsets,
        edge_types,
        angle_expansion,
):
    """Build a line graph over the concatenated heterogeneous bond nodes."""

    if edge_vec_all.size(0) == 0:
        line_edge_index = _empty_index(edge_vec_all)
        angle_attr = edge_vec_all.new_empty((0, angle_expansion.out_features))
        return line_edge_index, angle_attr

    valid_edges = _valid_edge_mask(edge_vec_all)
    outgoing = defaultdict(list)
    edge_targets = []

    for edge_type in edge_types:
        edge_index = edge_index_dict[edge_type]
        src_type, _, dst_type = edge_type
        offset = edge_offsets[edge_type]
        sources = edge_index[0].detach().cpu().tolist()
        targets = edge_index[1].detach().cpu().tolist()

        for local_edge, src in enumerate(sources):
            global_edge = offset + local_edge
            if valid_edges[global_edge]:
                outgoing[(src_type, int(src))].append(global_edge)
        for local_edge, dst in enumerate(targets):
            global_edge = offset + local_edge
            if valid_edges[global_edge]:
                edge_targets.append(((dst_type, int(dst)), global_edge))

    pair_src = []
    pair_dst = []
    for shared_node, first_edge in edge_targets:
        for second_edge in outgoing.get(shared_node, []):
            if first_edge == second_edge:
                continue
            pair_src.append(first_edge)
            pair_dst.append(second_edge)

    return _angle_features_from_pairs(
        pair_src,
        pair_dst,
        edge_vec_all,
        angle_expansion,
    )


class GatedGraphConv(nn.Module):
    """Official ALIGNN-style edge-gated convolution.

    The aggregation follows ALIGNN's gate-normalized update:
    sum(sigmoid(e_ij) * h_j) / (sum(sigmoid(e_ij)) + eps).
    """

    def __init__(self, channels, edge_dim, aggr="add", normalization="batchnorm"):
        super().__init__()
        self.channels = channels
        self.aggr = _normalize_aggr(aggr)
        self.src_gate = nn.Linear(channels, channels)
        self.dst_gate = nn.Linear(channels, channels)
        self.edge_gate = nn.Linear(edge_dim, channels)
        self.src_update = nn.Linear(channels, channels)
        self.dst_update = nn.Linear(channels, channels)
        # Keep the historical attribute names so ordinary ALIGNN checkpoints
        # retain their existing state-dict keys.
        self.bn_nodes = _make_feature_norm(channels, normalization)
        self.bn_edges = _make_feature_norm(channels, normalization)

    @staticmethod
    def _split_nodes(x):
        if isinstance(x, tuple):
            return x
        return x, x

    def forward(self, x, edge_index, edge_attr, size=None, return_edge_attr=False):
        x_src, x_dst = self._split_nodes(x)
        if x_dst.size(0) == 0 or edge_index.size(1) == 0:
            if return_edge_attr:
                return x_dst, edge_attr
            return x_dst

        src, dst = edge_index
        edge_update = (
            self.src_gate(x_src[src])
            + self.dst_gate(x_dst[dst])
            + self.edge_gate(edge_attr)
        )
        sigma = torch.sigmoid(edge_update)

        messages = self.dst_update(x_src)[src] * sigma
        out = x_dst.new_zeros((x_dst.size(0), self.channels))
        norm = x_dst.new_zeros((x_dst.size(0), self.channels))
        # AMP may produce half-precision linear outputs while residual roots
        # and their accumulation buffers remain float32. Keep accumulation in
        # the buffer dtype for both compatibility and numerical stability.
        out.index_add_(0, dst, messages.to(dtype=out.dtype))
        norm.index_add_(0, dst, sigma.to(dtype=norm.dtype))

        node_update = self.src_update(x_dst) + out / (norm + 1e-6)
        node_update = F.silu(self.bn_nodes(node_update))
        x_out = x_dst + node_update

        if not return_edge_attr:
            return x_out

        edge_update = F.silu(self.bn_edges(edge_update))
        if edge_attr.size(-1) == self.channels:
            edge_update = edge_attr + edge_update
        return x_out, edge_update


class HeteroRelationConv(nn.Module):
    """Relation-specific gated messages without a relation-local root update.

    Each relation returns the numerator and gate normalization required by the
    ALIGNN update. Callers normalize each relation independently and preserve
    the result in a separate slot for the node-type-specific fusion FFN.
    """

    def __init__(self, channels, edge_dim, normalization="batchnorm"):
        super().__init__()
        self.channels = channels
        self.src_gate = nn.Linear(channels, channels)
        self.dst_gate = nn.Linear(channels, channels)
        self.edge_gate = nn.Linear(edge_dim, channels)
        self.message_update = nn.Linear(channels, channels)
        self.bn_edges = _make_feature_norm(channels, normalization)

    @staticmethod
    def _split_nodes(x):
        if isinstance(x, tuple):
            return x
        return x, x

    def forward(self, x, edge_index, edge_attr, relation_adapter=None):
        x_src, x_dst = self._split_nodes(x)
        message_sum = x_dst.new_zeros((x_dst.size(0), self.channels))
        gate_sum = x_dst.new_zeros((x_dst.size(0), self.channels))
        if x_src.size(0) == 0 or x_dst.size(0) == 0 or edge_index.size(1) == 0:
            return message_sum, gate_sum, edge_attr

        src, dst = edge_index
        edge_update = (
            self.src_gate(x_src[src])
            + self.dst_gate(x_dst[dst])
            + self.edge_gate(edge_attr)
        )
        messages = self.message_update(x_src)[src]
        if relation_adapter is not None:
            gate_delta, message_delta = relation_adapter(torch.cat(
                [x_src[src], x_dst[dst], edge_attr], dim=-1,
            )).chunk(2, dim=-1)
            edge_update = edge_update + gate_delta
            messages = messages + message_delta
        sigma = torch.sigmoid(edge_update)
        messages = messages * sigma
        message_sum.index_add_(
            0,
            dst,
            messages.to(dtype=message_sum.dtype),
        )
        gate_sum.index_add_(0, dst, sigma.to(dtype=gate_sum.dtype))

        edge_update = F.silu(self.bn_edges(edge_update))
        if edge_attr.size(-1) == self.channels:
            edge_update = edge_attr + edge_update
        return message_sum, gate_sum, edge_update


class RelationResidualAdapter(nn.Module):
    """Small relation-specific corrections to gate logits and message values."""

    def __init__(self, channels, edge_dim, rank):
        super().__init__()
        self.down = nn.Linear(2 * channels + edge_dim, rank)
        self.up = nn.Linear(rank, 2 * channels)
        self.scale_logit = nn.Parameter(torch.tensor(-3.0))
        # The first forward is exactly the shared model. Gradients can train
        # the output projection immediately; the bottleneck follows thereafter.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs):
        return self.scale_logit.sigmoid() * self.up(F.silu(self.down(inputs)))


class SharedHeteroRelations(nn.Module):
    """One message core per layer, optionally with independent relation adapters."""

    def __init__(self, shared, relation_keys, mode, rank):
        super().__init__()
        self.shared = shared
        self.relation_keys = tuple(relation_keys)
        self.mode = mode
        # Adapters must not change initialization of the remaining model when
        # comparing shared and shared_residual under the same seed.
        with torch.random.fork_rng(devices=[]):
            self.adapters = nn.ModuleDict({
                key: RelationResidualAdapter(shared.channels, shared.edge_gate.in_features, rank)
                for key in self.relation_keys
            } if mode == 'shared_residual' else {})

    def forward(self, relation_key, x, edge_index, edge_attr):
        if relation_key not in self.relation_keys:
            raise KeyError(f'Unknown relation: {relation_key}')
        adapter = self.adapters[relation_key] if self.mode == 'shared_residual' else None
        return self.shared(x, edge_index, edge_attr, relation_adapter=adapter)


def _make_hetero_relation_convs(hidden_dim, edge_types, feature_norm, mode, rank):
    validate_alignn_hetero_relations(mode, rank)
    # Initialize in the legacy order to keep embeddings, node fusion and the
    # prediction head identical across relation modes at a fixed random seed.
    convs = nn.ModuleDict({
        key: HeteroRelationConv(hidden_dim, hidden_dim, normalization=feature_norm)
        for key in dict.fromkeys(_edge_parameter_key(edge_type) for edge_type in edge_types)
    })
    if mode == 'independent':
        return convs
    if not convs:
        raise ValueError('Shared hetero messages require at least one relation')
    # The other independently initialized cores are discarded, not registered.
    return SharedHeteroRelations(next(iter(convs.values())), tuple(convs), mode, rank)


class HeteroNodeUpdate(nn.Module):
    """Fuse relation slots and optionally normalize only the residual delta.

    Keeping ``x`` outside the normalization preserves the identity path.
    LayerNorm avoids batch-statistic noise for graphs with very few defect
    nodes, while SafeBatchNorm1d provides the directly comparable ablation.
    """

    def __init__(self, channels, num_relations, normalization="layernorm"):
        super().__init__()
        if isinstance(normalization, bool):
            normalization = "layernorm" if normalization else "none"
        normalization = str(normalization).lower()
        if normalization not in {"layernorm", "batchnorm", "none"}:
            raise ValueError(
                "normalization must be one of: layernorm, batchnorm, none"
            )
        self.normalization = normalization
        self.fusion = RelationFusionUpdate(channels, num_relations)
        self.layer_norm = (
            nn.LayerNorm(channels)
            if normalization == "layernorm"
            else nn.Identity()
        )
        self.batch_norm = (
            SafeBatchNorm1d(channels)
            if normalization == "batchnorm"
            else nn.Identity()
        )

    def forward(self, x, incoming_by_relation):
        fused = self.fusion(x, incoming_by_relation)
        delta = fused - x
        delta = self.layer_norm(delta)
        delta = self.batch_norm(delta)
        return x + delta


def _hetero_relation_update(
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        node_types,
        edge_types,
        relation_convs,
        node_updates,
):
    """Aggregate each relation independently, then fuse fixed relation slots."""
    incoming_by_edge_type = {}
    out_edge_attr = {}

    for edge_type in edge_types:
        src_type, _, dst_type = edge_type
        relation_key = _edge_parameter_key(edge_type)
        inputs = ((x_dict[src_type], x_dict[dst_type]),
                  edge_index_dict[edge_type], edge_attr_dict[edge_type])
        if isinstance(relation_convs, SharedHeteroRelations):
            message_sum, gate_sum, edge_update = relation_convs(relation_key, *inputs)
        else:
            message_sum, gate_sum, edge_update = relation_convs[relation_key](*inputs)
        incoming_by_edge_type[edge_type] = (
            message_sum / (gate_sum + 1e-6)
        )
        out_edge_attr[edge_type] = edge_update

    out_dict = {}
    node_type_order = {
        node_type: index for index, node_type in enumerate(node_types)
    }
    for node_type in node_types:
        incoming_edge_types = sorted(
            (
                edge_type
                for edge_type in edge_types
                if edge_type[2] == node_type
            ),
            key=lambda edge_type: node_type_order[edge_type[0]],
        )
        out_dict[node_type] = node_updates[node_type](
            x_dict[node_type],
            [
                incoming_by_edge_type[edge_type]
                for edge_type in incoming_edge_types
            ],
        )
    return out_dict, out_edge_attr


class GraphConvLayer(nn.Module):
    """Atom graph update block used after ALIGNN line-graph blocks."""

    def __init__(self, hidden_dim, vertex_aggregation="add", normalization="batchnorm"):
        super().__init__()
        self.atom_conv = GatedGraphConv(
            hidden_dim,
            hidden_dim,
            aggr=vertex_aggregation,
            normalization=normalization,
        )

    def forward(self, x, edge_index, edge_attr, size=None):
        return self.atom_conv(
            x,
            edge_index,
            edge_attr,
            size=size,
            return_edge_attr=True,
        )


class AtomTypeAttentionGatedGraphConv(MessagePassing):
    """Gated atom update with neighbor attention conditioned on atom type."""

    def __init__(self, channels, edge_dim, n_heads=4, aggr="add",
                 normalization="layernorm", legacy_residual_norm=False):
        super().__init__(aggr=_normalize_aggr(aggr))
        self.channels = channels
        self.n_heads = n_heads
        self.message_nn = nn.Sequential(
            nn.Linear(2 * channels + edge_dim, channels), nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.gate_nn = nn.Sequential(
            nn.Linear(2 * channels + edge_dim, channels), nn.SiLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.attn_nn = nn.Sequential(
            nn.Linear(4 * channels + edge_dim, 2 * n_heads), nn.SiLU(),
            nn.Linear(2 * n_heads, n_heads),
        )
        self.update_nn = nn.Sequential(
            nn.Linear(2 * channels, channels), nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.type_emb = nn.Embedding(2, channels)
        self.norm = _make_feature_norm(channels, normalization)
        self.legacy_residual_norm = bool(legacy_residual_norm)
        self._edge_index = None
        self._type_emb = None
        self._attention_weights = None

    def forward(self, x, edge_index, edge_attr, node_type=None, size=None):
        x_dst = x[1] if isinstance(x, tuple) else x
        if x_dst.size(0) == 0 or edge_index.size(1) == 0:
            return x_dst

        self._edge_index = edge_index
        if node_type is None:
            self._type_emb = None
        else:
            self._type_emb = self.type_emb(
                node_type.to(device=x_dst.device, dtype=torch.long).view(-1).clamp(0, 1)
            )
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)
        delta = self.update_nn(torch.cat([x_dst, out], dim=-1))
        if self.legacy_residual_norm:
            return self.norm(x_dst + delta)
        return x_dst + self.norm(delta)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        from torch_geometric.utils import softmax as pyg_softmax

        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.gate_nn(z) * self.message_nn(z)
        if self._type_emb is None:
            zeros = torch.zeros(z.size(0), 2 * self.channels, device=z.device, dtype=z.dtype)
            attn_input = torch.cat([z, zeros], dim=-1)
        else:
            edge_index = self._edge_index
            type_src = self._type_emb[edge_index[0]]
            type_dst = self._type_emb[edge_index[1]]
            attn_input = torch.cat([z, type_src, type_dst], dim=-1)
        alpha = pyg_softmax(self.attn_nn(attn_input), index, ptr, size_i)
        self._attention_weights = alpha.detach()
        return alpha.mean(dim=-1, keepdim=True) * msg

    def get_attention_weights(self):
        return self._attention_weights, self._edge_index


class ALIGNNLayer(nn.Module):
    """Old official ALIGNN block: atom graph update, then line graph update."""

    def __init__(self, hidden_dim, vertex_aggregation="add"):
        super().__init__()
        self.atom_conv = GatedGraphConv(hidden_dim, hidden_dim, aggr=vertex_aggregation)
        self.line_conv = GatedGraphConv(hidden_dim, hidden_dim, aggr=vertex_aggregation)

    def forward(self, x, edge_index, edge_attr, line_edge_index, angle_attr):
        x, edge_message = self.atom_conv(
            x,
            edge_index,
            edge_attr,
            return_edge_attr=True,
        )
        edge_attr, angle_attr = self.line_conv(
            edge_message,
            line_edge_index,
            angle_attr,
            return_edge_attr=True,
        )
        return x, edge_attr, angle_attr


class AttentionALIGNNLayer(nn.Module):
    """ALIGNN block with atom-type-aware attention on the atom graph."""

    def __init__(self, hidden_dim, angle_dim, n_heads=4,
                 vertex_aggregation="add", normalization="layernorm",
                 legacy_residual_norm=False):
        super().__init__()
        self.line_conv = GatedGraphConv(
            hidden_dim,
            hidden_dim,
            aggr=vertex_aggregation,
            normalization=normalization,
        )
        self.atom_conv = AtomTypeAttentionGatedGraphConv(
            hidden_dim,
            hidden_dim,
            n_heads=n_heads,
            aggr=vertex_aggregation,
            normalization=normalization,
            legacy_residual_norm=legacy_residual_norm,
        )

    def forward(self, x, edge_index, edge_attr, line_edge_index, angle_attr, node_type=None):
        edge_attr, angle_attr = self.line_conv(
            edge_attr,
            line_edge_index,
            angle_attr,
            return_edge_attr=True,
        )
        x = self.atom_conv(x, edge_index, edge_attr, node_type=node_type)
        return x, edge_attr, angle_attr

    def get_attention_weights(self):
        return self.atom_conv.get_attention_weights()


class DefiNetALIGNNLayer(nn.Module):
    """ALIGNN angle update followed by DeFiNet-style defect-aware atom update."""

    def __init__(self, hidden_dim, angle_dim, n_marker_types=2,
                 vertex_aggregation="add", normalization="layernorm",
                 legacy_residual_norm=False):
        super().__init__()
        self.line_conv = GatedGraphConv(
            hidden_dim,
            hidden_dim,
            aggr=vertex_aggregation,
            normalization=normalization,
        )
        self.atom_conv = DefectAwareGateConv(
            channels=hidden_dim,
            dim=hidden_dim,
            n_marker_types=n_marker_types,
            normalization=normalization,
            legacy_residual_norm=legacy_residual_norm,
        )

    def forward(self, x, edge_index, edge_attr, line_edge_index, angle_attr, defect_marker=None):
        edge_attr, angle_attr = self.line_conv(
            edge_attr,
            line_edge_index,
            angle_attr,
            return_edge_attr=True,
        )
        x = self.atom_conv(x, edge_index, edge_attr, defect_marker=defect_marker)
        return x, edge_attr, angle_attr

    def get_attention_weights(self):
        return self.atom_conv.get_attention_weights()


class HeteroALIGNNLayer(nn.Module):
    """Heterogeneous ALIGNN block for atom/defect node and edge types."""

    def __init__(self, hidden_dim, angle_dim, metadata,
                 vertex_aggregation="add", node_delta_norm="layernorm",
                 feature_norm="batchnorm", relation_mode="independent", relation_rank=8):
        super().__init__()
        self.node_types = tuple(metadata[0])
        self.edge_types = tuple(tuple(edge_type) for edge_type in metadata[1])
        self.line_conv = GatedGraphConv(
            hidden_dim, hidden_dim, aggr=vertex_aggregation,
            normalization=feature_norm,
        )
        self.atom_convs = _make_hetero_relation_convs(
            hidden_dim, self.edge_types, feature_norm, relation_mode, relation_rank,
        )
        self.node_updates = nn.ModuleDict({
            node_type: HeteroNodeUpdate(
                hidden_dim,
                sum(edge_type[2] == node_type for edge_type in self.edge_types),
                normalization=node_delta_norm,
            )
            for node_type in self.node_types
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict,
                line_edge_index, angle_attr, edge_offsets):
        edge_all = torch.cat(
            [edge_attr_dict[edge_type] for edge_type in self.edge_types],
            dim=0,
        )
        edge_all, angle_attr = self.line_conv(
            edge_all,
            line_edge_index,
            angle_attr,
            return_edge_attr=True,
        )

        angle_edge_attr_dict = {}
        for edge_type in self.edge_types:
            start = edge_offsets[edge_type]
            end = start + edge_attr_dict[edge_type].size(0)
            angle_edge_attr_dict[edge_type] = edge_all[start:end]

        out_dict, out_edge_attr = _hetero_relation_update(
            x_dict,
            edge_index_dict,
            angle_edge_attr_dict,
            self.node_types,
            self.edge_types,
            self.atom_convs,
            self.node_updates,
        )
        return out_dict, out_edge_attr, angle_attr


class HeteroGraphConvLayer(nn.Module):
    """Heterogeneous graph-conv block after ALIGNN line-graph blocks."""

    def __init__(self, hidden_dim, metadata, vertex_aggregation="add",
                 node_delta_norm="layernorm", feature_norm="batchnorm",
                 relation_mode="independent", relation_rank=8):
        super().__init__()
        self.node_types = tuple(metadata[0])
        self.edge_types = tuple(tuple(edge_type) for edge_type in metadata[1])
        self.atom_convs = _make_hetero_relation_convs(
            hidden_dim, self.edge_types, feature_norm, relation_mode, relation_rank,
        )
        self.node_updates = nn.ModuleDict({
            node_type: HeteroNodeUpdate(
                hidden_dim,
                sum(edge_type[2] == node_type for edge_type in self.edge_types),
                normalization=node_delta_norm,
            )
            for node_type in self.node_types
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        return _hetero_relation_update(
            x_dict,
            edge_index_dict,
            edge_attr_dict,
            self.node_types,
            self.edge_types,
            self.atom_convs,
            self.node_updates,
        )


class ALIGNN(nn.Module):
    """Homogeneous ALIGNN matching the old official ALIGNN architecture."""

    def __init__(
            self,
            node_input_shape,
            edge_input_shape,
            hidden_dim=64,
            n_blocks=3,
            gcn_blocks=4,
            angle_embed_size=40,
            vertex_aggregation="add",
            cutoff=8.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_embedding = MLPLayer(node_input_shape, hidden_dim)
        self.distance_expansion = RBFExpansion(0.0, cutoff, edge_input_shape)
        self.edge_embedding = _official_feature_embedding(edge_input_shape, hidden_dim)
        self.angle_expansion = AngleExpansion(angle_embed_size)
        self.angle_embedding = _official_feature_embedding(self.angle_expansion.out_features, hidden_dim)
        self.layers = nn.ModuleList([
            ALIGNNLayer(hidden_dim, vertex_aggregation=vertex_aggregation)
            for _ in range(n_blocks)
        ])
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, vertex_aggregation=vertex_aggregation)
            for _ in range(gcn_blocks)
        ])
        self.fc = nn.Linear(hidden_dim, 1)

    def _embed_edges(self, edge_attr, edge_vec):
        return _embed_distance_edges(
            edge_attr,
            edge_vec,
            self.distance_expansion,
            self.edge_embedding,
        )

    def forward(self, x, edge_index, edge_attr, batch, edge_vec=None):
        x = self.node_embedding(x.float())
        raw_edge_vec = edge_vec
        if edge_vec is None:
            edge_vec = edge_attr.new_zeros((edge_attr.size(0), 3))
        else:
            edge_vec = edge_vec.float()
        edge_attr = self._embed_edges(edge_attr, raw_edge_vec)
        line_edge_index, angle_attr = build_line_graph(edge_index, edge_vec, self.angle_expansion)
        angle_attr = self.angle_embedding(angle_attr.float())

        for layer in self.layers:
            x, edge_attr, angle_attr = layer(
                x,
                edge_index,
                edge_attr,
                line_edge_index,
                angle_attr,
            )
        for layer in self.gcn_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        num_graphs = _graph_count(batch=batch)
        node_pool = _pool_mean_or_zeros(x, batch, num_graphs, self.hidden_dim, x)
        return self.fc(node_pool)


class HyperALIGNN(nn.Module):
    """ALIGNN bond/angle updates interleaved with region hypergraph updates."""

    def __init__(
            self,
            node_input_shape,
            edge_input_shape,
            hidden_dim=64,
            n_blocks=3,
            gcn_blocks=4,
            angle_embed_size=40,
            n_heads=4,
            dropout=0.0,
            vertex_aggregation="add",
            cutoff=8.0,
            hypergraph_schema='per_defect_neighborhood_v2',
            hypergraph_pooling=None,
            hypergraph_updates=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.is_v3 = hypergraph_schema == 'defect_global_attention_v3'
        normalization = 'layernorm' if self.is_v3 else 'batchnorm'
        self.node_embedding = MLPLayer(node_input_shape, hidden_dim, normalization=normalization)
        self.distance_expansion = RBFExpansion(0.0, cutoff, edge_input_shape)
        self.edge_embedding = _official_feature_embedding(
            edge_input_shape,
            hidden_dim,
            normalization=normalization,
        )
        self.angle_expansion = AngleExpansion(angle_embed_size)
        self.angle_embedding = _official_feature_embedding(
            self.angle_expansion.out_features,
            hidden_dim,
            normalization=normalization,
        )
        self.layers = nn.ModuleList([
            AttentionALIGNNLayer(hidden_dim, self.angle_expansion.out_features,
                                n_heads=n_heads, vertex_aggregation=vertex_aggregation,
                                normalization=normalization)
            if self.is_v3 else ALIGNNLayer(hidden_dim, vertex_aggregation=vertex_aggregation)
            for _ in range(n_blocks)
        ])
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, vertex_aggregation=vertex_aggregation,
                           normalization=normalization)
            for _ in range(gcn_blocks)
        ])
        self.hypergraph = RegionHypergraphInteraction(
            hidden_dim,
            n_steps=n_blocks + gcn_blocks,
            heads=n_heads,
            dropout=dropout,
            schema=hypergraph_schema,
            pooling=hypergraph_pooling,
            updates=hypergraph_updates,
        )
        if self.is_v3 and not self.hypergraph.defect_mean:
            self.node_readout = AtomTypeGlobalAttentionReadout(hidden_dim)
        readout_dim = hidden_dim if self.hypergraph.defect_mean else 4 * hidden_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, 2 * hidden_dim), nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
            self,
            x,
            edge_index,
            edge_attr,
            batch,
            hyperedge_index,
            edge_vec=None,
            hyperedge_type=None,
            region_type=None,
    ):
        x = self.node_embedding(x.float())
        (
            num_graphs,
            num_hyperedges,
            hyperedge_index,
            hyperedge_type,
            region_type,
        ) = (
            self.hypergraph.normalize_inputs(
                x,
                hyperedge_index,
                batch=batch,
                hyperedge_type=hyperedge_type,
                region_type=region_type,
            )
        )
        x = self.hypergraph.add_region_features(x, region_type)

        raw_edge_vec = edge_vec
        if edge_vec is None:
            edge_vec = edge_attr.new_zeros((edge_attr.size(0), 3))
        else:
            edge_vec = edge_vec.float()
        edge_attr = _embed_distance_edges(
            edge_attr,
            raw_edge_vec,
            self.distance_expansion,
            self.edge_embedding,
        )
        line_edge_index, angle_attr = build_line_graph(
            edge_index,
            edge_vec,
            self.angle_expansion,
        )
        angle_attr = self.angle_embedding(angle_attr.float())

        step = 0
        for layer in self.layers:
            kwargs = {'node_type': region_type.eq(0).long()} if self.is_v3 else {}
            x, edge_attr, angle_attr = layer(
                x,
                edge_index,
                edge_attr,
                line_edge_index,
                angle_attr,
                **kwargs,
            )
            x = self.hypergraph.update(
                step,
                x,
                hyperedge_index,
                hyperedge_type,
                num_hyperedges,
            )
            step += 1
        for layer in self.gcn_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
            x = self.hypergraph.update(
                step,
                x,
                hyperedge_index,
                hyperedge_type,
                num_hyperedges,
            )
            step += 1

        region_pool = self.hypergraph.pool(
            x,
            hyperedge_index,
            hyperedge_type,
            batch,
            num_graphs,
            num_hyperedges,
        )
        if self.hypergraph.defect_mean:
            return self.readout(region_pool)
        if self.is_v3:
            global_pool = self.node_readout(x, batch, node_type=region_type.eq(0).long())
        else:
            global_pool = _pool_mean_or_zeros(x, batch, num_graphs, self.hidden_dim, x)
        return self.readout(torch.cat([global_pool, region_pool], dim=-1))


class AttentionALIGNN(nn.Module):
    """Homogeneous ALIGNN with atom-type-aware local and global attention."""

    def __init__(
            self,
            node_input_shape,
            edge_input_shape,
            hidden_dim=64,
            n_blocks=3,
            gcn_blocks=4,
            angle_embed_size=40,
            n_heads=4,
            vertex_aggregation="add",
            cutoff=8.0,
            normalization="layernorm",
            legacy_residual_norm=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_embedding = MLPLayer(
            node_input_shape,
            hidden_dim,
            normalization=normalization,
        )
        self.distance_expansion = RBFExpansion(0.0, cutoff, edge_input_shape)
        self.edge_embedding = _official_feature_embedding(
            edge_input_shape,
            hidden_dim,
            normalization=normalization,
        )
        self.angle_expansion = AngleExpansion(angle_embed_size)
        self.angle_embedding = _official_feature_embedding(
            self.angle_expansion.out_features,
            hidden_dim,
            normalization=normalization,
        )
        self.layers = nn.ModuleList([
            AttentionALIGNNLayer(
                hidden_dim,
                self.angle_expansion.out_features,
                n_heads=n_heads,
                vertex_aggregation=vertex_aggregation,
                normalization=normalization,
                legacy_residual_norm=legacy_residual_norm,
            )
            for _ in range(n_blocks)
        ])
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(
                hidden_dim,
                vertex_aggregation=vertex_aggregation,
                normalization=normalization,
            )
            for _ in range(gcn_blocks)
        ])
        self.node_readout = AtomTypeGlobalAttentionReadout(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, edge_vec=None, node_type=None):
        x = self.node_embedding(x.float())
        raw_edge_vec = edge_vec
        if edge_vec is None:
            edge_vec = edge_attr.new_zeros((edge_attr.size(0), 3))
        else:
            edge_vec = edge_vec.float()
        edge_attr = _embed_distance_edges(
            edge_attr,
            raw_edge_vec,
            self.distance_expansion,
            self.edge_embedding,
        )
        line_edge_index, angle_attr = build_line_graph(edge_index, edge_vec, self.angle_expansion)
        angle_attr = self.angle_embedding(angle_attr.float())
        if node_type is None:
            node_type = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            node_type = node_type.to(device=x.device, dtype=torch.long).view(-1)

        for layer in self.layers:
            x, edge_attr, angle_attr = layer(
                x, edge_index, edge_attr, line_edge_index, angle_attr, node_type=node_type
            )
        for layer in self.gcn_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        node_pool = self.node_readout(x, batch, node_type=node_type)
        return self.readout(node_pool)

    def get_all_attention_weights(self):
        results = []
        for i, layer in enumerate(self.layers):
            attn, edge_index = layer.get_attention_weights()
            if attn is not None:
                results.append((f'layer_{i}', attn, edge_index))
        attn = self.node_readout.get_attention_weights()
        if attn is not None:
            results.append(('global_readout', attn, None))
        return results


class DefiNetALIGNN(nn.Module):
    """DeFiNet-style ALIGNN gates without a graph-level virtual token."""

    def __init__(
            self,
            node_input_shape,
            edge_input_shape,
            hidden_dim=64,
            n_blocks=4,
            gcn_blocks=4,
            angle_embed_size=40,
            n_marker_types=2,
            vertex_aggregation="add",
            cutoff=8.0,
            normalization="layernorm",
            legacy_residual_norm=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_embedding = MLPLayer(
            node_input_shape,
            hidden_dim,
            normalization=normalization,
        )
        self.distance_expansion = RBFExpansion(0.0, cutoff, edge_input_shape)
        self.edge_embedding = _official_feature_embedding(
            edge_input_shape,
            hidden_dim,
            normalization=normalization,
        )
        self.angle_expansion = AngleExpansion(angle_embed_size)
        self.angle_embedding = _official_feature_embedding(
            self.angle_expansion.out_features,
            hidden_dim,
            normalization=normalization,
        )
        self.layers = nn.ModuleList([
            DefiNetALIGNNLayer(
                hidden_dim,
                self.angle_expansion.out_features,
                n_marker_types=n_marker_types,
                vertex_aggregation=vertex_aggregation,
                normalization=normalization,
                legacy_residual_norm=legacy_residual_norm,
            )
            for _ in range(n_blocks)
        ])
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(
                hidden_dim,
                vertex_aggregation=vertex_aggregation,
                normalization=normalization,
            )
            for _ in range(gcn_blocks)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch, edge_vec=None, defect_marker=None):
        x = self.node_embedding(x.float())
        raw_edge_vec = edge_vec
        if edge_vec is None:
            edge_vec = edge_attr.new_zeros((edge_attr.size(0), 3))
        else:
            edge_vec = edge_vec.float()
        edge_attr = _embed_distance_edges(
            edge_attr,
            raw_edge_vec,
            self.distance_expansion,
            self.edge_embedding,
        )
        line_edge_index, angle_attr = build_line_graph(edge_index, edge_vec, self.angle_expansion)
        angle_attr = self.angle_embedding(angle_attr.float())
        if defect_marker is None:
            defect_marker = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            defect_marker = defect_marker.to(device=x.device, dtype=torch.long).view(-1)

        for layer in self.layers:
            x, edge_attr, angle_attr = layer(
                x, edge_index, edge_attr, line_edge_index, angle_attr,
                defect_marker=defect_marker,
            )
        for layer in self.gcn_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        num_graphs = _graph_count(batch=batch)
        node_pool = _pool_mean_or_zeros(x, batch, num_graphs, self.hidden_dim, x)
        return self.readout(node_pool)

    def get_all_attention_weights(self):
        results = []
        for i, layer in enumerate(self.layers):
            attn, edge_index = layer.get_attention_weights()
            if attn is not None:
                results.append((f'layer_{i}', attn, edge_index))
        return results


class HeteroALIGNN(nn.Module):
    """Heterogeneous ALIGNN with configurable feature norms and defect pooling.

    Constructor defaults preserve pre-existing checkpoints. New training
    configs select LayerNorm, actual-defect mean pooling and shared message
    cores with small relation-specific residual adapters.
    """

    def __init__(
            self,
            node_input_shape,
            edge_input_shape,
            metadata,
            hidden_dim=64,
            n_blocks=3,
            gcn_blocks=4,
            angle_embed_size=40,
            vertex_aggregation="add",
            fixed_pooling=False,
            node_delta_norm="layernorm",
            cutoff=8.0,
            feature_norm="batchnorm",
            pooling="type_mean",
            relation_mode="independent",
            relation_rank=8,
    ):
        super().__init__()
        self.node_types = tuple(metadata[0])
        self.edge_types = tuple(tuple(edge_type) for edge_type in metadata[1])
        self.hidden_dim = hidden_dim
        self.fixed_pooling = fixed_pooling
        self.node_delta_norm = node_delta_norm
        if feature_norm not in {"batchnorm", "layernorm"}:
            raise ValueError("feature_norm must be batchnorm or layernorm")
        if pooling not in {"type_mean", "defect_mean"}:
            raise ValueError("pooling must be type_mean or defect_mean")
        self.feature_norm = feature_norm
        self.pooling = pooling
        validate_alignn_hetero_relations(relation_mode, relation_rank)
        self.relation_mode = relation_mode
        self.relation_rank = relation_rank
        self.node_embedding = nn.ModuleDict({
            node_type: MLPLayer(node_input_shape, hidden_dim, normalization=feature_norm)
            for node_type in self.node_types
        })
        self.distance_expansion = RBFExpansion(0.0, cutoff, edge_input_shape)
        self.edge_embedding = nn.ModuleDict({
            parameter_key: _official_feature_embedding(
                edge_input_shape, hidden_dim, normalization=feature_norm,
            )
            for parameter_key in dict.fromkeys(
                _edge_parameter_key(edge_type) for edge_type in self.edge_types
            )
        })
        self.angle_expansion = AngleExpansion(angle_embed_size)
        self.angle_embedding = _official_feature_embedding(
            self.angle_expansion.out_features, hidden_dim, normalization=feature_norm,
        )
        self.layers = nn.ModuleList([
            HeteroALIGNNLayer(
                hidden_dim,
                self.angle_expansion.out_features,
                metadata,
                vertex_aggregation=vertex_aggregation,
                node_delta_norm=node_delta_norm,
                feature_norm=feature_norm,
                relation_mode=relation_mode,
                relation_rank=relation_rank,
            )
            for _ in range(n_blocks)
        ])
        self.gcn_layers = nn.ModuleList([
            HeteroGraphConvLayer(
                hidden_dim,
                metadata,
                vertex_aggregation=vertex_aggregation,
                node_delta_norm=node_delta_norm,
                feature_norm=feature_norm,
                relation_mode=relation_mode,
                relation_rank=relation_rank,
            )
            for _ in range(gcn_blocks)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim if pooling == "defect_mean" else 2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _line_graph_inputs(self, edge_index_dict, edge_attr_dict, edge_vec_dict):
        edge_offsets = {}
        edge_vec_parts = []
        offset = 0

        for edge_type in self.edge_types:
            edge_attr = edge_attr_dict[edge_type]
            edge_vec = edge_vec_dict.get(edge_type)
            if edge_vec is None:
                edge_vec = edge_attr.new_zeros((edge_attr.size(0), 3))
            edge_offsets[edge_type] = offset
            edge_vec_parts.append(edge_vec.float())
            offset += int(edge_attr.size(0))

        reference = next(iter(edge_attr_dict.values()))
        if offset == 0:
            line_edge_index = _empty_index(reference)
            angle_attr = reference.new_empty((0, self.angle_expansion.out_features))
        else:
            edge_vec_all = torch.cat(edge_vec_parts, dim=0)
            line_edge_index, angle_attr = build_hetero_line_graph(
                edge_index_dict,
                edge_vec_all,
                edge_offsets,
                self.edge_types,
                self.angle_expansion,
            )
        return (
            line_edge_index,
            self.angle_embedding(angle_attr.float()),
            edge_offsets,
        )

    @staticmethod
    def _pool_type_for_node_store(pool_type, node_type, x):
        if pool_type is not None:
            return pool_type.to(device=x.device, dtype=torch.long).view(-1)
        default_value = 1 if node_type == "defect" else 0
        return torch.full((x.size(0),), default_value, dtype=torch.long, device=x.device)

    def _pool_fixed_type(self, x_dict, batch_dict, pool_type_dict, target_type,
                         num_graphs, reference, require_nonempty=False):
        features = []
        batches = []
        pool_type_dict = {} if pool_type_dict is None else pool_type_dict
        for node_type, x in x_dict.items():
            if x is None or x.size(0) == 0:
                continue
            batch = batch_dict.get(node_type)
            if batch is None or batch.numel() == 0:
                continue
            pool_type = self._pool_type_for_node_store(
                pool_type_dict.get(node_type), node_type, x
            )
            if pool_type.numel() != x.size(0) or not torch.all((pool_type == 0) | (pool_type == 1)):
                raise ValueError("pool_type must contain one binary marker per node")
            mask = pool_type.eq(target_type)
            if torch.count_nonzero(mask) == 0:
                continue
            features.append(x[mask])
            batches.append(batch[mask])
        if not features:
            if require_nonempty:
                raise ValueError("defect_mean pooling requires at least one defect per graph")
            return _pool_mean_or_zeros(None, None, num_graphs, self.hidden_dim, reference)
        pooled_batch = torch.cat(batches, dim=0)
        if require_nonempty and torch.any(torch.bincount(pooled_batch, minlength=num_graphs) == 0):
            raise ValueError("defect_mean pooling requires at least one defect per graph")
        return _pool_mean_or_zeros(
            torch.cat(features, dim=0),
            pooled_batch,
            num_graphs,
            self.hidden_dim,
            reference,
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict,
                edge_vec_dict=None, state=None, pool_type=None):
        edge_vec_dict = {} if edge_vec_dict is None else edge_vec_dict
        x_dict = {
            node_type: self.node_embedding[node_type](x_dict[node_type].float())
            for node_type in self.node_types
        }
        edge_attr_dict = {
            edge_type: _embed_distance_edges(
                edge_attr_dict[edge_type],
                edge_vec_dict.get(edge_type),
                self.distance_expansion,
                self.edge_embedding[_edge_parameter_key(edge_type)],
            )
            for edge_type in self.edge_types
        }
        line_edge_index, angle_attr, edge_offsets = self._line_graph_inputs(
            edge_index_dict,
            edge_attr_dict,
            edge_vec_dict,
        )

        for layer in self.layers:
            x_dict, edge_attr_dict, angle_attr = layer(
                x_dict,
                edge_index_dict,
                edge_attr_dict,
                line_edge_index,
                angle_attr,
                edge_offsets,
            )
        for layer in self.gcn_layers:
            x_dict, edge_attr_dict = layer(x_dict, edge_index_dict, edge_attr_dict)

        reference = next(value for value in x_dict.values() if value is not None)
        num_graphs = _graph_count(batch_dict=batch_dict, state=state)
        if self.pooling == "defect_mean":
            # pool_type retains the actual defects when r > 0 also marks
            # surrounding pristine sites as members of the defect node store.
            defect_pool = self._pool_fixed_type(
                x_dict, batch_dict, pool_type, 1, num_graphs, reference,
                require_nonempty=True,
            )
            return self.readout(defect_pool)
        if self.fixed_pooling:
            atom_pool = self._pool_fixed_type(x_dict, batch_dict, pool_type, 0, num_graphs, reference)
            defect_pool = self._pool_fixed_type(x_dict, batch_dict, pool_type, 1, num_graphs, reference)
        else:
            atom_pool = _pool_mean_or_zeros(
                x_dict.get("atom"), batch_dict.get("atom"), num_graphs, self.hidden_dim, reference
            )
            defect_pool = _pool_mean_or_zeros(
                x_dict.get("defect"), batch_dict.get("defect"), num_graphs, self.hidden_dim, reference
            )

        return self.readout(torch.cat([atom_pool, defect_pool], dim=-1))
