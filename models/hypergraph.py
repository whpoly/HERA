"""Region hypergraph network for crystalline defect property prediction."""

import torch
from torch import nn
from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import scatter


class RegionHypergraphBlock(nn.Module):
    """One residual node -> hyperedge -> node message-passing block."""

    def __init__(self, hidden_dim, heads=4, dropout=0.0):
        super().__init__()
        self.conv = HypergraphConv(
            hidden_dim,
            hidden_dim,
            use_attention=True,
            attention_mode="node",
            heads=heads,
            concat=False,
            dropout=dropout,
        )
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, hyperedge_index, hyperedge_attr, num_hyperedges):
        delta = self.conv(
            x,
            hyperedge_index,
            hyperedge_attr=hyperedge_attr,
            num_edges=num_hyperedges,
        )
        x = x + self.message_norm(delta)
        return x + self.ffn_norm(self.ffn(x))


class RegionHypergraphInteraction(nn.Module):
    """Reusable three-region hypergraph updates for a physical GNN backbone."""

    NUM_REGION_TYPES = 3

    def __init__(self, hidden_dim, n_steps, heads=4, dropout=0.0):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if heads < 1:
            raise ValueError("heads must be >= 1")
        self.hidden_dim = int(hidden_dim)
        self.region_embedding = nn.Embedding(self.NUM_REGION_TYPES, hidden_dim)
        self.blocks = nn.ModuleList([
            RegionHypergraphBlock(hidden_dim, heads=heads, dropout=dropout)
            for _ in range(n_steps)
        ])

    @classmethod
    def graph_count(cls, batch=None, state=None):
        if batch is not None and batch.numel() > 0:
            return int(batch.max().item()) + 1
        if state is not None and state.numel() > 0:
            return int(state.reshape(-1, state.shape[-1]).shape[0])
        return 1

    def normalize_inputs(
            self,
            x,
            hyperedge_index,
            batch=None,
            state=None,
            hyperedge_type=None,
            region_type=None,
    ):
        num_graphs = self.graph_count(batch=batch, state=state)
        num_hyperedges = num_graphs * self.NUM_REGION_TYPES
        if hyperedge_type is None:
            hyperedge_type = torch.arange(
                num_hyperedges,
                device=x.device,
                dtype=torch.long,
            ) % self.NUM_REGION_TYPES
        else:
            hyperedge_type = hyperedge_type.to(
                device=x.device,
                dtype=torch.long,
            ).view(-1)
        if hyperedge_type.numel() != num_hyperedges:
            raise ValueError(
                f"Expected {num_hyperedges} hyperedge types for {num_graphs} "
                f"graphs, got {hyperedge_type.numel()}"
            )
        if torch.any((hyperedge_type < 0) | (hyperedge_type >= self.NUM_REGION_TYPES)):
            raise ValueError("hyperedge_type values must be in [0, 2]")

        if region_type is None:
            region_type = torch.full(
                (x.size(0),),
                -1,
                dtype=torch.long,
                device=x.device,
            )
            region_type[hyperedge_index[0]] = hyperedge_type[hyperedge_index[1]]
        else:
            region_type = region_type.to(device=x.device, dtype=torch.long).view(-1)
        if region_type.numel() != x.size(0):
            raise ValueError(
                f"Expected one region type per node ({x.size(0)}), "
                f"got {region_type.numel()}"
            )
        if torch.any((region_type < 0) | (region_type >= self.NUM_REGION_TYPES)):
            raise ValueError("region_type values must be in [0, 2] for every node")
        return num_graphs, num_hyperedges, hyperedge_type, region_type

    def add_region_features(self, x, region_type):
        return x + self.region_embedding(region_type)

    def update(self, step, x, hyperedge_index, hyperedge_type, num_hyperedges):
        return self.blocks[step](
            x,
            hyperedge_index,
            self.region_embedding(hyperedge_type),
            num_hyperedges,
        )

    def pool(self, x, hyperedge_index, num_graphs):
        node_ids, hyperedge_ids = hyperedge_index
        return scatter(
            x[node_ids],
            hyperedge_ids,
            dim=0,
            dim_size=num_graphs * self.NUM_REGION_TYPES,
            reduce="mean",
        ).reshape(num_graphs, self.NUM_REGION_TYPES * self.hidden_dim)


class RegionHypergraphNet(nn.Module):
    """Hypergraph model with three fixed semantic hyperedges per crystal.

    The hyperedges represent, in order:

    0. defect atoms;
    1. pristine atoms within the configured defect-neighbor radius;
    2. all remaining pristine atoms.

    Every atom belongs to exactly one of the three hyperedges. Empty region
    hyperedges are retained so batched graphs always have a stable layout.
    """

    NUM_REGION_TYPES = RegionHypergraphInteraction.NUM_REGION_TYPES

    def __init__(
            self,
            node_input_shape,
            hidden_dim=64,
            n_blocks=3,
            heads=4,
            state_input_shape=2,
            dropout=0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.node_embedding = nn.Sequential(
            nn.Linear(node_input_shape, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.hypergraph = RegionHypergraphInteraction(
            hidden_dim,
            n_steps=n_blocks,
            heads=heads,
            dropout=dropout,
        )
        self.state_embedding = nn.Sequential(
            nn.Linear(state_input_shape, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear((self.NUM_REGION_TYPES + 1) * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
            self,
            x,
            hyperedge_index,
            batch,
            state=None,
            hyperedge_type=None,
            region_type=None,
    ):
        num_graphs, num_hyperedges, hyperedge_type, region_type = (
            self.hypergraph.normalize_inputs(
                x,
                hyperedge_index,
                batch=batch,
                state=state,
                hyperedge_type=hyperedge_type,
                region_type=region_type,
            )
        )
        x = self.hypergraph.add_region_features(self.node_embedding(x), region_type)
        for step in range(len(self.hypergraph.blocks)):
            x = self.hypergraph.update(
                step,
                x,
                hyperedge_index,
                hyperedge_type,
                num_hyperedges,
            )
        region_pool = self.hypergraph.pool(x, hyperedge_index, num_graphs)

        if state is None:
            state = x.new_zeros((num_graphs, self.state_embedding[0].in_features))
        else:
            state = state.to(device=x.device, dtype=x.dtype).reshape(num_graphs, -1)
        state_pool = self.state_embedding(state)
        return self.readout(torch.cat([region_pool, state_pool], dim=-1))
