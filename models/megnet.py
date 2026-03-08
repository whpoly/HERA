"""MEGNet model variants: MEGNet (homogeneous), HeteroMEGNet, AttentionMEGNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.aggr import Set2Set

from .modules import (
    ATOMIC_NUMBERS,
    ShiftedSoftplus,
    MegnetModule,
    HeteroMegnetLayer,
    AtomTypeAttentionMegnetModule,
)


class MEGNet(nn.Module):
    """Homogeneous MEGNet for full / sparse modes."""

    def __init__(self,
                 edge_input_shape,
                 node_input_shape,
                 state_input_shape,
                 node_embedding_size=16,
                 embedding_size=32,
                 n_blocks=3,
                 vertex_aggregation="mean",
                 global_aggregation="mean",
                 ):
        super().__init__()
        self.embedded = node_input_shape is None
        if self.embedded:
            node_input_shape = node_embedding_size
        self.emb = nn.Embedding(ATOMIC_NUMBERS, node_embedding_size)
        self.m1 = MegnetModule(
            edge_input_shape, node_input_shape, state_input_shape,
            inner_skip=True, embed_size=embedding_size,
            vertex_aggregation=vertex_aggregation,
            global_aggregation=global_aggregation,
        )
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks - 1):
            self.blocks.append(MegnetModule(
                embedding_size, embedding_size, embedding_size,
                embed_size=embedding_size,
                vertex_aggregation=vertex_aggregation,
                global_aggregation=global_aggregation,
            ))

        self.se = Set2Set(embedding_size, 1)
        self.sv = Set2Set(embedding_size, 1)
        self.hiddens = nn.Sequential(
            nn.Linear(5 * embedding_size, embedding_size), ShiftedSoftplus(),
            nn.Linear(embedding_size, embedding_size // 2), ShiftedSoftplus(),
            nn.Linear(embedding_size // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch):
        if self.embedded:
            x = self.emb(x.long()).squeeze()
        else:
            x = x.float()

        x, edge_attr, state = self.m1(x, edge_index, edge_attr, state, batch, bond_batch)
        for block in self.blocks:
            x, edge_attr, state = block(x, edge_index, edge_attr, state, batch, bond_batch)

        x = self.sv(x, batch)
        edge_attr = self.se(edge_attr, bond_batch)
        tmp_shape = x.shape[0] - edge_attr.shape[0]
        edge_attr = F.pad(edge_attr, (0, 0, 0, tmp_shape), value=0.0)
        tmp = torch.cat((x, edge_attr, state), 1)
        return self.hiddens(tmp)


class HeteroMEGNet(nn.Module):
    """Heterogeneous MEGNet for hetero mode."""

    def __init__(self,
                 edge_input_shape,
                 node_input_shape,
                 state_input_shape,
                 metadata,
                 node_embedding_size=16,
                 embedding_size=32,
                 n_blocks=3,
                 vertex_aggregation="mean",
                 global_aggregation="mean",
                 ):
        super().__init__()
        self.node_type = metadata[0]
        self.edge_type = metadata[1]
        self.embedded = node_input_shape is None
        if self.embedded:
            node_input_shape = node_embedding_size
            self.emb = nn.Embedding(ATOMIC_NUMBERS, node_embedding_size)
        self.m1 = HeteroMegnetLayer(
            edge_input_shape, node_input_shape, state_input_shape,
            metadata, inner_skip=True, embedding_size=embedding_size,
            vertex_aggregation=vertex_aggregation,
            global_aggregation=global_aggregation,
        )
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks - 1):
            self.blocks.append(HeteroMegnetLayer(
                embedding_size, embedding_size, embedding_size,
                metadata, embedding_size=embedding_size,
                vertex_aggregation=vertex_aggregation,
                global_aggregation=global_aggregation,
            ))

        self.se = Set2Set(embedding_size, 1)
        self.sv = Set2Set(embedding_size, 1)
        self.sv_2 = Set2Set(embedding_size, 1)
        self.hiddens = nn.Sequential(
            nn.Linear(7 * embedding_size, 2 * embedding_size), ShiftedSoftplus(),
            nn.Linear(2 * embedding_size, embedding_size), ShiftedSoftplus(),
            nn.Linear(embedding_size, 1),
        )

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch):
        if self.embedded:
            x = {k: self.emb(v.long()).squeeze() for k, v in x.items()}
        else:
            x = {k: v.float() for k, v in x.items()}

        x, edge_attr, state = self.m1(x, edge_index, edge_attr, state, batch, bond_batch)
        for block in self.blocks:
            x, edge_attr, state = block(x, edge_index, edge_attr, state, batch, bond_batch)

        x_atom = self.sv(x['atom'], batch['atom'])
        x_defect = self.sv_2(x['defect'], batch['defect'])

        edge_attr = torch.cat([edge_attr[self.edge_type[i]] for i in range(len(self.edge_type))], dim=0)
        bond_batch = torch.cat([bond_batch[self.edge_type[i]] for i in range(len(self.edge_type))], dim=0)
        edge_attr = self.se(edge_attr, bond_batch)

        tmp_shape = x_atom.shape[0] - edge_attr.shape[0]
        edge_attr = F.pad(edge_attr, (0, 0, 0, tmp_shape), value=0.0)
        tmp = torch.cat((x_atom, x_defect, edge_attr, state), 1)
        return self.hiddens(tmp)


class AttentionMEGNet(nn.Module):
    """MEGNet with atom-type-aware multi-head attention."""

    def __init__(self,
                 edge_input_shape,
                 node_input_shape,
                 state_input_shape,
                 node_embedding_size=16,
                 embedding_size=32,
                 n_blocks=3,
                 n_heads=4,
                 vertex_aggregation="mean",
                 global_aggregation="mean",
                 ):
        super().__init__()
        self.embedded = node_input_shape is None
        if self.embedded:
            node_input_shape = node_embedding_size
        self.emb = nn.Embedding(ATOMIC_NUMBERS, node_embedding_size)
        self.n_heads = n_heads

        self.m1 = AtomTypeAttentionMegnetModule(
            edge_input_shape, node_input_shape, state_input_shape,
            inner_skip=True, embed_size=embedding_size, n_heads=n_heads,
            vertex_aggregation=vertex_aggregation,
            global_aggregation=global_aggregation,
        )
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks - 1):
            self.blocks.append(AtomTypeAttentionMegnetModule(
                embedding_size, embedding_size, embedding_size,
                embed_size=embedding_size, n_heads=n_heads,
                vertex_aggregation=vertex_aggregation,
                global_aggregation=global_aggregation,
            ))

        self.se = Set2Set(embedding_size, 1)
        self.sv = Set2Set(embedding_size, 1)
        self.hiddens = nn.Sequential(
            nn.Linear(5 * embedding_size, embedding_size), ShiftedSoftplus(),
            nn.Linear(embedding_size, embedding_size // 2), ShiftedSoftplus(),
            nn.Linear(embedding_size // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr, state, batch, bond_batch, node_type=None):
        if self.embedded:
            x = self.emb(x.long()).squeeze()
        else:
            x = x.float()

        x, edge_attr, state = self.m1(x, edge_index, edge_attr, state, batch, bond_batch, node_type=node_type)
        for block in self.blocks:
            x, edge_attr, state = block(x, edge_index, edge_attr, state, batch, bond_batch, node_type=node_type)

        x = self.sv(x, batch)
        edge_attr = self.se(edge_attr, bond_batch)
        tmp_shape = x.shape[0] - edge_attr.shape[0]
        edge_attr = F.pad(edge_attr, (0, 0, 0, tmp_shape), value=0.0)
        tmp = torch.cat((x, edge_attr, state), 1)
        return self.hiddens(tmp)

    def get_all_attention_weights(self):
        results = []
        attn, ei = self.m1.get_attention_weights()
        if attn is not None:
            results.append(('block_0', attn, ei))
        for i, block in enumerate(self.blocks):
            attn, ei = block.get_attention_weights()
            if attn is not None:
                results.append((f'block_{i+1}', attn, ei))
        return results
