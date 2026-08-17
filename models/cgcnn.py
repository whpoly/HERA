"""CGCNN model variants: CGCNN (homogeneous), CrystalGraphConvNet, Heterocgcnn, AttentionCGCNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, MeanAggregation

from .modules import (
    AttentionCGConv,
    AtomTypeGlobalAttentionReadout,
    DefectAwareGateConv,
    RelationFusionUpdate,
    ShiftedSoftplus,
)


class CGCNN(nn.Module):
    """Homogeneous CGCNN for full and WAS modes."""

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        super().__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([
            CGConv(channels=(atom_fea_len, atom_fea_len), dim=nbr_fea_len, batch_norm=True)
            for _ in range(n_conv)
        ])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.pooling = MeanAggregation()

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 8)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        atom_fea = self.embedding(x)
        for conv_func in self.convs:
            atom_fea = conv_func(x=atom_fea, edge_index=edge_index, edge_attr=edge_attr)

        crys_fea = self.pooling(atom_fea, batch)
        crys_fea = self.conv_to_fc(F.softplus(crys_fea))
        crys_fea = F.softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        return self.fc_out(crys_fea)


class CrystalGraphConvNet(nn.Module):
    """CGCNN backbone (convolutions only, no readout) — used inside Heterocgcnn."""

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        super().__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([
            CGConv(channels=(atom_fea_len, atom_fea_len), dim=nbr_fea_len, batch_norm=True)
            for _ in range(n_conv)
        ])

    def forward(self, x, edge_index, edge_attr, batch):
        atom_fea = self.embedding(x)
        for conv_func in self.convs:
            atom_fea = conv_func(x=atom_fea, edge_index=edge_index, edge_attr=edge_attr)
        return atom_fea


class CGCNNRelationConv(nn.Module):
    """CGCNN message function with mean aggregation inside one relation."""

    def __init__(self, channels, edge_dim):
        super().__init__()
        message_input_dim = 2 * channels + edge_dim
        self.channels = channels
        self.lin_filter = nn.Linear(message_input_dim, channels)
        self.lin_core = nn.Linear(message_input_dim, channels)

    def forward(self, x, edge_index, edge_attr):
        x_src, x_dst = x
        out = x_dst.new_zeros((x_dst.size(0), self.channels))
        if x_src.size(0) == 0 or x_dst.size(0) == 0 or edge_index.size(1) == 0:
            return out

        src, dst = edge_index
        z = torch.cat((x_dst[dst], x_src[src], edge_attr), dim=-1)
        messages = torch.sigmoid(self.lin_filter(z)) * F.softplus(self.lin_core(z))
        out.index_add_(0, dst, messages)

        counts = out.new_zeros((x_dst.size(0), 1))
        counts.index_add_(
            0,
            dst,
            out.new_ones((dst.size(0), 1)),
        )
        return out / counts.clamp_min(1)


class HeteroCrystalGraphConvNet(nn.Module):
    """Relation-preserving heterogeneous CGCNN backbone.

    Incoming relations are mean-aggregated independently. A node-type-specific
    FFN then fuses the root state and the fixed relation slots.
    """

    def __init__(
            self,
            orig_atom_fea_len,
            nbr_fea_len,
            metadata,
            atom_fea_len=64,
            n_conv=3,
    ):
        super().__init__()
        self.node_types = tuple(metadata[0])
        self.edge_types = tuple(tuple(edge_type) for edge_type in metadata[1])
        node_type_order = {
            node_type: index for index, node_type in enumerate(self.node_types)
        }
        self.incoming_edge_types = {
            node_type: sorted(
                (
                    edge_type
                    for edge_type in self.edge_types
                    if edge_type[2] == node_type
                ),
                key=lambda edge_type: node_type_order[edge_type[0]],
            )
            for node_type in self.node_types
        }
        self.embedding = nn.ModuleDict({
            node_type: nn.Linear(orig_atom_fea_len, atom_fea_len)
            for node_type in self.node_types
        })
        self.convs = nn.ModuleList([
            nn.ModuleDict({
                "__".join(edge_type): CGCNNRelationConv(
                    atom_fea_len,
                    nbr_fea_len,
                )
                for edge_type in self.edge_types
            })
            for _ in range(n_conv)
        ])
        self.node_updates = nn.ModuleList([
            nn.ModuleDict({
                node_type: RelationFusionUpdate(
                    atom_fea_len,
                    len(self.incoming_edge_types[node_type]),
                    normalization="layernorm",
                )
                for node_type in self.node_types
            })
            for _ in range(n_conv)
        ])

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = {
            node_type: self.embedding[node_type](x[node_type].float())
            for node_type in self.node_types
        }
        for relation_convs, node_updates in zip(self.convs, self.node_updates):
            incoming_by_edge_type = {}
            for edge_type in self.edge_types:
                src_type, _, dst_type = edge_type
                incoming_by_edge_type[edge_type] = (
                    relation_convs["__".join(edge_type)](
                        (x[src_type], x[dst_type]),
                        edge_index[edge_type],
                        edge_attr[edge_type],
                    )
                )
            x = {
                node_type: node_updates[node_type](
                    x[node_type],
                    [
                        incoming_by_edge_type[edge_type]
                        for edge_type in self.incoming_edge_types[node_type]
                    ],
                )
                for node_type in self.node_types
            }
        return x


class Heterocgcnn(nn.Module):
    """Heterogeneous CGCNN wrapper for hetero mode."""

    def __init__(self, base_model, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, fixed_pooling=False):
        super().__init__()
        self.classification = classification
        self.base_model = base_model
        self.atom_fea_len = atom_fea_len
        self.fixed_pooling = fixed_pooling
        self.conv_to_fc = nn.Linear(2 * atom_fea_len, h_fea_len)
        self.pooling = MeanAggregation()

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 8)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def _pool_node_type(self, features, batch, dim_size, reference):
        if features is None or batch is None or features.size(0) == 0:
            return reference.new_zeros((dim_size, self.atom_fea_len))
        try:
            return self.pooling(features, batch, dim_size=dim_size)
        except TypeError:
            pooled = self.pooling(features, batch)
            if pooled.size(0) == dim_size:
                return pooled
            padded = reference.new_zeros((dim_size, pooled.size(1)))
            padded[:pooled.size(0)] = pooled[:dim_size]
            return padded

    @staticmethod
    def _num_graphs(batch):
        counts = [int(value.max().item()) + 1 for value in batch.values() if value.numel() > 0]
        return max(counts) if counts else 1

    @staticmethod
    def _pool_type_for_node_store(pool_type, node_type, x):
        if pool_type is not None:
            return pool_type.to(device=x.device, dtype=torch.long).view(-1)
        default_value = 1 if node_type == 'defect' else 0
        return torch.full((x.size(0),), default_value, dtype=torch.long, device=x.device)

    def _pool_fixed_type(self, atom_fea, batch, pool_type, target_type, num_graphs, reference):
        features = []
        batches = []
        pool_type = {} if pool_type is None else pool_type
        for node_type, features_by_type in atom_fea.items():
            if features_by_type is None or features_by_type.size(0) == 0:
                continue
            batch_by_type = batch.get(node_type)
            if batch_by_type is None or batch_by_type.numel() == 0:
                continue
            node_pool_type = self._pool_type_for_node_store(
                pool_type.get(node_type), node_type, features_by_type
            )
            mask = node_pool_type.eq(target_type)
            if torch.count_nonzero(mask) == 0:
                continue
            features.append(features_by_type[mask])
            batches.append(batch_by_type[mask])
        if not features:
            return self._pool_node_type(None, None, num_graphs, reference)
        return self._pool_node_type(
            torch.cat(features, dim=0),
            torch.cat(batches, dim=0),
            num_graphs,
            reference,
        )

    def forward(self, x, edge_index, edge_attr, batch, pool_type=None):
        atom_fea = self.base_model(x, edge_index, edge_attr, batch)
        reference = next(value for value in atom_fea.values() if value is not None)
        num_graphs = self._num_graphs(batch)
        if self.fixed_pooling:
            defect_pool = self._pool_fixed_type(atom_fea, batch, pool_type, 1, num_graphs, reference)
            atom_pool = self._pool_fixed_type(atom_fea, batch, pool_type, 0, num_graphs, reference)
        else:
            defect_fea = atom_fea['defect']
            defect_batch = batch['defect']
            defect_pool = self._pool_node_type(defect_fea, defect_batch, num_graphs, reference)
            atom_pool = self._pool_node_type(atom_fea.get('atom'), batch.get('atom'), num_graphs, reference)
        crys_fea = torch.cat((
            defect_pool,
            atom_pool,
        ), 1)

        crys_fea = self.conv_to_fc(F.softplus(crys_fea))
        crys_fea = F.softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        return self.fc_out(crys_fea)


class AttentionCGCNN(nn.Module):
    """CGCNN with atom-type-aware multi-head attention convolutions."""

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 n_heads=4, classification=False):
        super().__init__()
        self.classification = classification
        self.n_heads = n_heads

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([
            AttentionCGConv(channels=atom_fea_len, dim=nbr_fea_len,
                            n_heads=n_heads, batch_norm=True)
            for _ in range(n_conv)
        ])

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.pooling = AtomTypeGlobalAttentionReadout(atom_fea_len)

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 8)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, x, edge_index, edge_attr, batch, node_type=None):
        atom_fea = self.embedding(x)
        for conv_func in self.convs:
            atom_fea = conv_func(x=atom_fea, edge_index=edge_index, edge_attr=edge_attr, node_type=node_type)

        crys_fea = self.pooling(atom_fea, batch, node_type=node_type)
        crys_fea = self.conv_to_fc(F.softplus(crys_fea))
        crys_fea = F.softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        return self.fc_out(crys_fea)

    def get_all_attention_weights(self):
        results = []
        for i, conv in enumerate(self.convs):
            attn, ei = conv.get_attention_weights()
            if attn is not None:
                results.append((f'conv_{i}', attn, ei))
        attn = self.pooling.get_attention_weights()
        if attn is not None:
            results.append(('global_readout', attn, None))
        return results


class DefiNet(nn.Module):
    """Scalar-property adapter of DeFiNet defect-aware message passing.

    The paper predicts relaxed coordinates with scalar/vector/coordinate
    triplets. This class keeps the repository's scalar-regression readout
    while using DeFiNet's marker-pair gated scalar message passing and a
    lightweight scalar global node.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=4, h_fea_len=128, n_h=1,
                 n_marker_types=2, classification=False):
        super().__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.global_seed = nn.Parameter(torch.zeros(1, atom_fea_len))
        self.global_distribute = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * atom_fea_len, atom_fea_len), ShiftedSoftplus(),
                nn.Linear(atom_fea_len, atom_fea_len),
            )
            for _ in range(n_conv)
        ])
        self.convs = nn.ModuleList([
            DefectAwareGateConv(
                channels=atom_fea_len,
                dim=nbr_fea_len,
                n_marker_types=n_marker_types,
                batch_norm=True,
            )
            for _ in range(n_conv)
        ])
        self.global_aggregate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * atom_fea_len, atom_fea_len), ShiftedSoftplus(),
                nn.Linear(atom_fea_len, atom_fea_len),
            )
            for _ in range(n_conv)
        ])
        self.pooling = MeanAggregation()
        self.conv_to_fc = nn.Linear(2 * atom_fea_len, h_fea_len)

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 8)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, x, edge_index, edge_attr, batch, defect_marker=None):
        atom_fea = self.embedding(x)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        global_fea = self.global_seed.expand(num_graphs, -1)

        for distribute, conv_func, aggregate in zip(
                self.global_distribute, self.convs, self.global_aggregate):
            atom_fea = atom_fea + distribute(torch.cat([atom_fea, global_fea[batch]], dim=-1))
            atom_fea = conv_func(
                x=atom_fea,
                edge_index=edge_index,
                edge_attr=edge_attr,
                defect_marker=defect_marker,
            )
            pooled = self.pooling(atom_fea, batch)
            global_fea = global_fea + aggregate(torch.cat([pooled, global_fea], dim=-1))

        crys_fea = torch.cat([self.pooling(atom_fea, batch), global_fea], dim=-1)
        crys_fea = self.conv_to_fc(F.softplus(crys_fea))
        crys_fea = F.softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        return self.fc_out(crys_fea)

    def get_all_attention_weights(self):
        results = []
        for i, conv in enumerate(self.convs):
            attn, ei = conv.get_attention_weights()
            if attn is not None:
                results.append((f'conv_{i}', attn, ei))
        return results
