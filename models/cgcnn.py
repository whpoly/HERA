"""CGCNN model variants: CGCNN (homogeneous), CrystalGraphConvNet, Heterocgcnn, AttentionCGCNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, MeanAggregation

from .modules import AttentionCGConv, DefectAwareGateConv, ShiftedSoftplus


class CGCNN(nn.Module):
    """Homogeneous CGCNN for full / sparse modes."""

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


class Heterocgcnn(nn.Module):
    """Heterogeneous CGCNN wrapper for hetero mode."""

    def __init__(self, base_model, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        super().__init__()
        self.classification = classification
        self.base_model = base_model
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

    def forward(self, x, edge_index, edge_attr, batch):
        atom_fea = self.base_model(x, edge_index, edge_attr, batch)
        crys_fea = torch.cat((
            self.pooling(atom_fea['defect'], batch['defect']),
            self.pooling(atom_fea['atom'], batch['atom']),
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
        self.pooling = MeanAggregation()

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

        crys_fea = self.pooling(atom_fea, batch)
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
