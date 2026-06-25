"""Shared neural network building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    aggr,
)

ATOMIC_NUMBERS = 95


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()
        self.shift = nn.Parameter(torch.log(torch.tensor([2.])), requires_grad=False)

    def forward(self, x):
        return self.sp(x) - self.shift


# ------------------------------------------------------------------ #
#  MEGNet message-passing module (homogeneous)
# ------------------------------------------------------------------ #

class MegnetModule(MessagePassing):
    def __init__(self,
                 edge_input_shape,
                 node_input_shape,
                 state_input_shape,
                 inner_skip=False,
                 embed_size=32,
                 vertex_aggregation="mean",
                 global_aggregation="mean",
                 ):
        if vertex_aggregation == "lin":
            vertex_aggregation = aggr.MultiAggregation(
                ['mean', 'sum', 'max'],
                mode='proj',
                mode_kwargs={
                    'in_channels': embed_size,
                    'out_channels': embed_size,
                },
            )
        super().__init__(aggr=vertex_aggregation)

        if global_aggregation == "mean":
            self.global_aggregation = global_mean_pool
        elif global_aggregation == "sum":
            self.global_aggregation = global_add_pool
        elif global_aggregation == "max":
            self.global_aggregation = global_max_pool
        else:
            raise ValueError("Unknown global aggregation type")

        self.inner_skip = inner_skip
        self.phi_e = nn.Sequential(
            nn.Linear(4 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )
        self.phi_u = nn.Sequential(
            nn.Linear(3 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )
        self.phi_v = nn.Sequential(
            nn.Linear(3 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )
        self.preprocess_e = nn.Sequential(
            nn.Linear(edge_input_shape, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )
        self.preprocess_v = nn.Sequential(
            nn.Linear(node_input_shape, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )
        self.preprocess_u = nn.Sequential(
            nn.Linear(state_input_shape, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch):
        if not self.inner_skip:
            x_skip = x
            edge_attr_skip = edge_attr
            state_skip = state
            x = self.preprocess_v(x)
            edge_attr = self.preprocess_e(edge_attr)
            state = self.preprocess_u(state)
        else:
            x = self.preprocess_v(x)
            edge_attr = self.preprocess_e(edge_attr)
            state = self.preprocess_u(state)
            x_skip = x
            edge_attr_skip = edge_attr
            state_skip = state

        if torch.numel(bond_batch) > 0:
            edge_attr = self.edge_updater(
                edge_index=edge_index, x=x, edge_attr=edge_attr,
                state=state, bond_batch=bond_batch
            )
        x = self.propagate(
            edge_index=edge_index, x=x, edge_attr=edge_attr,
            state=state, batch=batch
        )
        u_v = self.global_aggregation(x, batch)
        u_e = self.global_aggregation(edge_attr, bond_batch, batch.max().item() + 1)
        state = self.phi_u(torch.cat((u_e, u_v, state), 1))
        return x + x_skip, edge_attr + edge_attr_skip, state + state_skip

    def message(self, x_i, x_j, edge_attr):
        return edge_attr

    def update(self, inputs, x, state, batch):
        return self.phi_v(torch.cat((inputs, x, state[batch, :]), 1))

    def edge_update(self, x_i, x_j, edge_attr, state, bond_batch):
        return self.phi_e(torch.cat((x_i, x_j, edge_attr, state[bond_batch, :]), 1))


# ------------------------------------------------------------------ #
#  Heterogeneous MEGNet layer
# ------------------------------------------------------------------ #

class HeteroMegnetLayer(nn.Module):
    def __init__(self,
                 edge_input_shape,
                 node_input_shape,
                 state_input_shape,
                 metadata,
                 node_embedding_size=16,
                 embedding_size=32,
                 vertex_aggregation="mean",
                 global_aggregation="mean",
                 inner_skip=False,
                 ):
        super().__init__()
        self.edge_types = metadata[1]
        self.embedding_size = embedding_size
        self.megnets = nn.ModuleDict()
        for etype in self.edge_types:
            self.megnets['__'.join(etype)] = MegnetModule(
                edge_input_shape=edge_input_shape,
                node_input_shape=node_input_shape,
                state_input_shape=state_input_shape,
                embed_size=embedding_size,
                vertex_aggregation=vertex_aggregation,
                global_aggregation=global_aggregation,
                inner_skip=inner_skip,
            )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, state, batch_dict, bond_batch_dict):
        out_dict = {ntype: [] for ntype in x_dict}
        state_outs = []
        edge_out_dict = {}
        for etype in self.edge_types:
            k = '__'.join(etype)
            src, rel, dst = etype
            if (
                    x_dict[src].size(0) == 0
                    or x_dict[dst].size(0) == 0
                    or edge_index_dict[etype].size(1) == 0
            ):
                edge_out_dict[etype] = edge_attr_dict[etype].new_empty((0, self.embedding_size))
                continue
            if src != dst:
                src_count = x_dict[src].size(0)
                dst_count = x_dict[dst].size(0)
                x_input = torch.cat([x_dict[src], x_dict[dst]], dim=0)
                batch = torch.cat([batch_dict[src], batch_dict[dst]], dim=0)
                edge_index = edge_index_dict[etype].clone()
                edge_index[1] = edge_index[1] + src_count
                dst_slice = slice(src_count, src_count + dst_count)
            else:
                x_input = x_dict[src]
                batch = batch_dict[src]
                edge_index = edge_index_dict[etype]
                dst_slice = slice(None)

            edge_attr = edge_attr_dict[etype]
            bond_batch = bond_batch_dict[etype]
            x_out, edge_attr_out, state_out = self.megnets[k](
                x_input, edge_index, edge_attr, state, batch, bond_batch
            )
            x_dst = x_out[dst_slice, :]
            out_dict[dst].append(x_dst)
            edge_out_dict[etype] = edge_attr_out
            state_outs.append(state_out)

        agg_dict = {}
        for ntype, outs in out_dict.items():
            if len(outs) == 0:
                agg_dict[ntype] = x_dict[ntype]
            elif len(outs) == 1:
                agg_dict[ntype] = outs[0]
            else:
                agg_dict[ntype] = torch.stack(outs, dim=0).mean(dim=0)

        if len(state_outs) == 1:
            state_new = state_outs[0]
        elif len(state_outs) == 0:
            state_new = state
        else:
            state_new = torch.stack(state_outs, dim=0).mean(dim=0)
        return agg_dict, edge_out_dict, state_new


# ------------------------------------------------------------------ #
#  Attention-based MEGNet module (homogeneous graph with attention)
# ------------------------------------------------------------------ #

class AtomTypeAttentionMegnetModule(MessagePassing):
    def __init__(self,
                 edge_input_shape,
                 node_input_shape,
                 state_input_shape,
                 inner_skip=False,
                 embed_size=32,
                 n_heads=4,
                 vertex_aggregation="mean",
                 global_aggregation="mean",
                 ):
        if vertex_aggregation == "lin":
            vertex_aggregation = aggr.MultiAggregation(
                ['mean', 'sum', 'max'],
                mode='proj',
                mode_kwargs={
                    'in_channels': embed_size,
                    'out_channels': embed_size,
                },
            )
        super().__init__(aggr=vertex_aggregation)

        if global_aggregation == "mean":
            self.global_aggregation = global_mean_pool
        elif global_aggregation == "sum":
            self.global_aggregation = global_add_pool
        elif global_aggregation == "max":
            self.global_aggregation = global_max_pool
        else:
            raise ValueError("Unknown global aggregation type")

        self.inner_skip = inner_skip
        self.n_heads = n_heads
        self.embed_size = embed_size

        # Type embedding: 0=host, 1=defect
        self.type_emb = nn.Embedding(2, embed_size)
        self.attn_type_src = nn.Linear(embed_size, n_heads, bias=False)
        self.attn_type_dst = nn.Linear(embed_size, n_heads, bias=False)

        self.attn_src = nn.Linear(embed_size, n_heads, bias=False)
        self.attn_dst = nn.Linear(embed_size, n_heads, bias=False)
        self.attn_edge = nn.Linear(embed_size, n_heads, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self._attention_weights = None
        self._edge_index = None
        self._type_emb = None

        self.phi_e = nn.Sequential(
            nn.Linear(4 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )
        self.phi_u = nn.Sequential(
            nn.Linear(3 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )
        self.phi_v = nn.Sequential(
            nn.Linear(3 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )
        self.preprocess_e = nn.Sequential(
            nn.Linear(edge_input_shape, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )
        self.preprocess_v = nn.Sequential(
            nn.Linear(node_input_shape, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )
        self.preprocess_u = nn.Sequential(
            nn.Linear(state_input_shape, 2 * embed_size), ShiftedSoftplus(),
            nn.Linear(2 * embed_size, embed_size), ShiftedSoftplus(),
        )

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch, node_type=None):
        # Compute type embeddings if node_type is provided
        if node_type is not None:
            self._type_emb = self.type_emb(node_type)
        else:
            self._type_emb = None

        if not self.inner_skip:
            x_skip = x
            edge_attr_skip = edge_attr
            state_skip = state
            x = self.preprocess_v(x)
            edge_attr = self.preprocess_e(edge_attr)
            state = self.preprocess_u(state)
        else:
            x = self.preprocess_v(x)
            edge_attr = self.preprocess_e(edge_attr)
            state = self.preprocess_u(state)
            x_skip = x
            edge_attr_skip = edge_attr
            state_skip = state

        if torch.numel(bond_batch) > 0:
            edge_attr = self.edge_updater(
                edge_index=edge_index, x=x, edge_attr=edge_attr,
                state=state, bond_batch=bond_batch
            )

        self._edge_index = edge_index
        x = self.propagate(
            edge_index=edge_index, x=x, edge_attr=edge_attr,
            state=state, batch=batch
        )
        u_v = self.global_aggregation(x, batch)
        u_e = self.global_aggregation(edge_attr, bond_batch, batch.max().item() + 1)
        state = self.phi_u(torch.cat((u_e, u_v, state), 1))
        return x + x_skip, edge_attr + edge_attr_skip, state + state_skip

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        from torch_geometric.utils import softmax as pyg_softmax
        alpha_src = self.attn_src(x_j)
        alpha_dst = self.attn_dst(x_i)
        alpha_edge = self.attn_edge(edge_attr)
        alpha = alpha_src + alpha_dst + alpha_edge
        # Add type-based attention if node_type was provided
        if self._type_emb is not None:
            edge_index = self._edge_index
            alpha_type_src = self.attn_type_src(self._type_emb[edge_index[0]])
            alpha_type_dst = self.attn_type_dst(self._type_emb[edge_index[1]])
            alpha = alpha + alpha_type_src + alpha_type_dst
        alpha = self.leaky_relu(alpha)
        alpha = pyg_softmax(alpha, index, ptr, size_i)
        self._attention_weights = alpha.detach()
        attn = alpha.mean(dim=-1, keepdim=True)
        return attn * edge_attr

    def update(self, inputs, x, state, batch):
        return self.phi_v(torch.cat((inputs, x, state[batch, :]), 1))

    def edge_update(self, x_i, x_j, edge_attr, state, bond_batch):
        return self.phi_e(torch.cat((x_i, x_j, edge_attr, state[bond_batch, :]), 1))

    def get_attention_weights(self):
        return self._attention_weights, self._edge_index


# ------------------------------------------------------------------ #
#  Attention-based CGConv
# ------------------------------------------------------------------ #

class AttentionCGConv(MessagePassing):
    def __init__(self, channels, dim, n_heads=4, batch_norm=True):
        super().__init__(aggr='add')
        self.channels = channels
        self.dim = dim
        self.n_heads = n_heads

        # Type embedding: 0=host, 1=defect
        self.type_emb = nn.Embedding(2, channels)

        in_size = 2 * channels + dim
        self.lin_f = nn.Linear(in_size, channels)
        self.lin_s = nn.Linear(in_size, channels)
        # Attention now includes 2 extra type embeddings (src + dst)
        attn_in_size = in_size + 2 * channels
        self.attn_nn = nn.Sequential(
            nn.Linear(attn_in_size, n_heads * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(n_heads * 2, n_heads),
        )
        self.bn = nn.BatchNorm1d(channels) if batch_norm else nn.Identity()

        self._attention_weights = None
        self._edge_index = None
        self._type_emb = None

    def forward(self, x, edge_index, edge_attr, node_type=None):
        self._edge_index = edge_index
        if node_type is not None:
            self._type_emb = self.type_emb(node_type)
        else:
            self._type_emb = None
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.bn(out)
        out = out + x
        return out

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        from torch_geometric.utils import softmax as pyg_softmax
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
        # Add type embeddings to attention input
        if self._type_emb is not None:
            edge_index = self._edge_index
            t_src = self._type_emb[edge_index[0]]
            t_dst = self._type_emb[edge_index[1]]
            z_attn = torch.cat([z, t_src, t_dst], dim=-1)
        else:
            z_attn = torch.cat([z, torch.zeros(z.size(0), 2 * self.channels, device=z.device)], dim=-1)
        alpha = self.attn_nn(z_attn)
        alpha = pyg_softmax(alpha, index, ptr, size_i)
        self._attention_weights = alpha.detach()
        attn = alpha.mean(dim=-1, keepdim=True)
        return attn * msg

    def get_attention_weights(self):
        return self._attention_weights, self._edge_index


# ------------------------------------------------------------------ #
#  Atom-type-aware global attention readout
# ------------------------------------------------------------------ #

class AtomTypeGlobalAttentionReadout(nn.Module):
    """Graph-level attention readout conditioned on atom type.

    Local attention layers decide which neighboring bonds are important for
    each center atom. This readout learns which atoms are important for the
    whole-structure prediction, similar in spirit to GATGNN's global
    attention layer, while also conditioning on pristine/defect node labels.
    """

    def __init__(self, channels, n_node_types=2, hidden_channels=None, use_context=True):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.channels = channels
        self.use_context = use_context
        self.type_emb = nn.Embedding(n_node_types, channels)
        attn_in_size = 3 * channels if use_context else 2 * channels
        self.attn_nn = nn.Sequential(
            nn.Linear(attn_in_size, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, 1),
        )
        self._attention_weights = None

    def forward(self, x, batch, node_type=None):
        from torch_geometric.utils import softmax as pyg_softmax

        if node_type is None:
            node_type = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        node_type = node_type.to(device=x.device, dtype=torch.long).view(-1)
        type_emb = self.type_emb(node_type.clamp(0, self.type_emb.num_embeddings - 1))

        attn_inputs = [x, type_emb]
        if self.use_context:
            graph_context = global_mean_pool(x, batch)
            attn_inputs.append(graph_context[batch])

        scores = self.attn_nn(torch.cat(attn_inputs, dim=-1))
        alpha = pyg_softmax(scores, batch)
        self._attention_weights = alpha.detach()
        return global_add_pool(alpha * x, batch)

    def get_attention_weights(self):
        return self._attention_weights


# ------------------------------------------------------------------ #
#  DeFiNet-style defect-aware gated convolution (scalar branch)
# ------------------------------------------------------------------ #

class DefectAwareGateConv(MessagePassing):
    """Scalar DeFiNet-style message passing with marker-pair gates.

    This implements the scalar part of the DeFiNet defect-aware message:
    each neighbor message is re-weighted by distance features and by a
    learnable gate of the marker pair (m_i, m_j). Unlike the existing
    attention layers above, this layer does not apply neighbor softmax.
    """

    def __init__(self, channels, dim, n_marker_types=2, batch_norm=True):
        super().__init__(aggr='add')
        self.channels = channels
        self.dim = dim
        self.n_marker_types = n_marker_types

        self.marker_emb = nn.Embedding(n_marker_types, channels)
        self.phi_h = nn.Sequential(
            nn.Linear(channels, channels), ShiftedSoftplus(),
            nn.Linear(channels, channels),
        )
        self.lambda_h = nn.Sequential(
            nn.Linear(dim, channels), ShiftedSoftplus(),
            nn.Linear(channels, channels),
        )
        self.gamma_h = nn.Sequential(
            nn.Linear(channels, channels), ShiftedSoftplus(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.update_nn = nn.Sequential(
            nn.Linear(2 * channels, channels), ShiftedSoftplus(),
            nn.Linear(channels, channels),
        )
        self.bn = nn.BatchNorm1d(channels) if batch_norm else nn.Identity()

        self._edge_index = None
        self._markers = None
        self._gate_weights = None

    def forward(self, x, edge_index, edge_attr, defect_marker=None):
        self._edge_index = edge_index
        if defect_marker is None:
            defect_marker = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        defect_marker = defect_marker.to(device=x.device, dtype=torch.long).view(-1)
        self._markers = defect_marker.clamp(0, self.n_marker_types - 1)

        msg = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        out = x + self.update_nn(torch.cat([x, msg], dim=-1))
        return self.bn(out)

    def message(self, x_j, edge_attr):
        src, dst = self._edge_index
        marker_pair = self.marker_emb(self._markers[src]) + self.marker_emb(self._markers[dst])
        gate = self.gamma_h(marker_pair)
        self._gate_weights = gate.detach()
        return self.phi_h(x_j) * self.lambda_h(edge_attr) * gate

    def get_attention_weights(self):
        return self._gate_weights, self._edge_index
