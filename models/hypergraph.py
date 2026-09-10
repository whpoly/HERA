"""Region hypergraph network for crystalline defect property prediction."""

import torch
from torch import nn
from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import scatter, softmax

from ..config.defaults import (
    HYPERGRAPH_SCHEMA, LEGACY_HYPERGRAPH_SCHEMA, HYPERGRAPH_SCHEMAS,
    resolve_hypergraph_pooling,
)


def infer_region_type(num_nodes, hyperedge_index, hyperedge_type):
    """Leave nodes without incidences as far pristine, never as defects."""
    node_ids, edge_ids = hyperedge_index
    regions = node_ids.new_full((num_nodes,), 2)
    return regions.scatter_reduce(
        0, node_ids, hyperedge_type[edge_ids], reduce='amin', include_self=True,
    )


def hyperedge_context(x, hyperedge_index, hyperedge_type, num_edges):
    node_ids, edge_ids = hyperedge_index
    regions = infer_region_type(x.size(0), hyperedge_index, hyperedge_type)
    center_mask = regions[node_ids].eq(0)
    centers = scatter(x[node_ids[center_mask]], edge_ids[center_mask], dim=0,
                      dim_size=num_edges, reduce='mean')
    means = scatter(x[node_ids], edge_ids, dim=0, dim_size=num_edges, reduce='mean')
    return centers, means, regions


class DefectHypergraphBlock(nn.Module):
    """Dynamic node -> hyperedge -> node attention with separate softmaxes.

    A local edge is queried by its defect center and current environment;
    the global defect edge is queried by the defect ensemble. There is no
    additional inverse-cardinality factor after either softmax.
    """

    def __init__(self, hidden_dim, heads=4, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = (hidden_dim + heads - 1) // heads
        inner = heads * self.head_dim
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_query = nn.Linear(3 * hidden_dim, inner)
        self.node_key = nn.Linear(hidden_dim, inner)
        self.node_value = nn.Linear(hidden_dim, inner)
        self.edge_output = nn.Linear(inner, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_query = nn.Linear(hidden_dim, inner)
        self.edge_key = nn.Linear(hidden_dim, inner)
        self.edge_value = nn.Linear(hidden_dim, inner)
        self.node_output = nn.Linear(inner, hidden_dim, bias=False)
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.message_gate = nn.Parameter(torch.tensor(-2.0))
        self.ffn_gate = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(dropout)
        self._attention_weights = None

    def _heads(self, tensor):
        return tensor.reshape(-1, self.heads, self.head_dim)

    def forward(self, x, hyperedge_index, hyperedge_attr, num_hyperedges,
                hyperedge_type):
        node_ids, edge_ids = hyperedge_index
        z = self.node_norm(x)
        centers, means, _ = hyperedge_context(
            z, hyperedge_index, hyperedge_type, num_hyperedges,
        )
        q = self._heads(self.edge_query(torch.cat([centers, means, hyperedge_attr], -1)))
        k, v = self._heads(self.node_key(z)), self._heads(self.node_value(z))
        scores = (q[edge_ids] * k[node_ids]).sum(-1) / self.head_dim ** 0.5
        alpha = softmax(scores, edge_ids, num_nodes=num_hyperedges)
        edges = scatter(self.dropout(alpha).unsqueeze(-1) * v[node_ids], edge_ids,
                        dim=0, dim_size=num_hyperedges, reduce='sum')
        edges = self.edge_norm(self.edge_output(edges.flatten(1)) + centers + hyperedge_attr)

        q = self._heads(self.node_query(z))
        k, v = self._heads(self.edge_key(edges)), self._heads(self.edge_value(edges))
        scores = (q[node_ids] * k[edge_ids]).sum(-1) / self.head_dim ** 0.5
        beta = softmax(scores, node_ids, num_nodes=x.size(0))
        delta = scatter(self.dropout(beta).unsqueeze(-1) * v[edge_ids], node_ids,
                        dim=0, dim_size=x.size(0), reduce='sum')
        # Even learned biases must not update far nodes through this branch.
        active = scatter(x.new_ones((node_ids.numel(), 1)), node_ids, dim=0,
                         dim_size=x.size(0), reduce='sum').gt(0).to(x.dtype)
        x = x + active * self.message_gate.sigmoid() * self.message_norm(
            self.node_output(delta.flatten(1)),
        )
        x = x + active * self.ffn_gate.sigmoid() * self.ffn_norm(self.ffn(x))
        self._attention_weights = (alpha.detach(), beta.detach())
        return x


class DefectHierarchicalReadout(nn.Module):
    """Separate defect, local-environment and unique-pristine attention pools.

    Local nodes are pooled conditioned on their center before local edges are
    pooled conditioned on the global defect representation. Three bounded
    structural fractions retain concentration/overlap information lost by
    softmax. No raw atom-count multiplier assumes an extensive target.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.type_embedding = nn.Embedding(2, hidden_dim)
        self.node_score = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.local_score = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.host_score = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.fraction_embedding = nn.Linear(3, 3 * hidden_dim, bias=False)
        nn.init.zeros_(self.fraction_embedding.weight)
        self._attention_weights = None

    def forward(self, x, hyperedge_index, hyperedge_type, batch,
                num_graphs, num_hyperedges):
        node_ids, edge_ids = hyperedge_index
        centers, means, regions = hyperedge_context(
            x, hyperedge_index, hyperedge_type, num_hyperedges,
        )
        edge_graph = scatter(batch[node_ids], edge_ids, dim=0,
                             dim_size=num_hyperedges, reduce='min')
        scores = self.node_score(torch.cat([
            x[node_ids], centers[edge_ids], means[edge_ids],
            self.type_embedding(hyperedge_type)[edge_ids],
        ], -1))
        alpha = softmax(scores, edge_ids, num_nodes=num_hyperedges)
        edge_pool = scatter(alpha * x[node_ids], edge_ids, dim=0,
                            dim_size=num_hyperedges, reduce='sum')
        core = hyperedge_type.eq(0)
        defect_pool = scatter(edge_pool[core], edge_graph[core], dim=0,
                              dim_size=num_graphs, reduce='mean')
        local = hyperedge_type.eq(1)
        local_graph = edge_graph[local]
        local_scores = self.local_score(torch.cat([
            edge_pool[local], defect_pool[local_graph],
        ], -1))
        local_alpha = softmax(local_scores, local_graph, num_nodes=num_graphs)
        local_pool = scatter(local_alpha * edge_pool[local], local_graph, dim=0,
                             dim_size=num_graphs, reduce='sum')
        host = regions.ne(0)
        host_graph = batch[host]
        host_scores = self.host_score(torch.cat([x[host], defect_pool[host_graph]], -1))
        host_alpha = softmax(host_scores, host_graph, num_nodes=num_graphs)
        host_pool = scatter(host_alpha * x[host], host_graph, dim=0,
                            dim_size=num_graphs, reduce='sum')

        local_degree = scatter(local[edge_ids].to(x.dtype), node_ids, dim=0,
                               dim_size=x.size(0), reduce='sum')
        counts = scatter(x.new_ones(x.size(0)), batch, dim=0,
                         dim_size=num_graphs, reduce='sum').clamp_min(1)
        fractions = torch.stack([
            scatter(mask.to(x.dtype), batch, dim=0, dim_size=num_graphs, reduce='sum') / counts
            for mask in (regions.eq(0), regions.eq(1), host & local_degree.gt(1))
        ], -1)
        self._attention_weights = {
            'node_to_edge': alpha.detach(), 'local_edges': local_alpha.detach(),
            'pristine_nodes': host_alpha.detach(), 'fractions': fractions.detach(),
        }
        return torch.cat([defect_pool, local_pool, host_pool], -1) + self.fraction_embedding(fractions)


class DefectMeanReadout(nn.Module):
    """Mean each graph's final defect representations, counting nodes once.

    Physical and hypergraph propagation have already incorporated the host
    environment and other defects. This readout does not pool incidences,
    local edges, pristine nodes, or raw defect counts into the prediction.
    The subsequent graph MLP acts on the mean representation.
    """

    def forward(self, x, hyperedge_index, hyperedge_type, batch,
                num_graphs, num_hyperedges):
        regions = infer_region_type(x.size(0), hyperedge_index, hyperedge_type)
        defect = regions.eq(0)
        counts = scatter(defect.to(x.dtype), batch, dim=0,
                         dim_size=num_graphs, reduce='sum')
        if torch.any(counts.eq(0)):
            raise ValueError('Defect mean pooling requires at least one defect per graph')
        return scatter(x[defect], batch[defect], dim=0,
                       dim_size=num_graphs, reduce='mean')


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
    """Reusable per-defect hypergraph updates for a physical GNN backbone."""

    NUM_REGION_TYPES = 3
    DEFECT_CORE = 0
    LOCAL_NEIGHBORHOOD = 1
    FAR_FIELD = 2

    def __init__(self, hidden_dim, n_steps, heads=4, dropout=0.0,
                 schema=LEGACY_HYPERGRAPH_SCHEMA, pooling=None):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if heads < 1:
            raise ValueError("heads must be >= 1")
        self.hidden_dim = int(hidden_dim)
        if schema not in HYPERGRAPH_SCHEMAS:
            raise ValueError(f"Unknown hypergraph schema: {schema}")
        self.schema = schema
        self.is_v3 = schema == HYPERGRAPH_SCHEMA
        self.pooling_mode = resolve_hypergraph_pooling(schema, pooling)
        self.defect_mean = self.pooling_mode == 'defect_mean'
        self.output_dim = hidden_dim if self.defect_mean else 3 * hidden_dim
        self.region_embedding = nn.Embedding(self.NUM_REGION_TYPES, hidden_dim)
        block_type = DefectHypergraphBlock if self.is_v3 else RegionHypergraphBlock
        self.blocks = nn.ModuleList([
            block_type(hidden_dim, heads=heads, dropout=dropout)
            for _ in range(n_steps)
        ])
        if self.is_v3:
            self.readout = (DefectMeanReadout() if self.defect_mean
                            else DefectHierarchicalReadout(hidden_dim))

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
        if hyperedge_type is None:
            raise ValueError(
                "hyperedge_type is required for variable per-defect hypergraphs"
            )
        hyperedge_type = hyperedge_type.to(
            device=x.device,
            dtype=torch.long,
        ).view(-1)
        num_hyperedges = int(hyperedge_type.numel())
        if num_hyperedges < 1:
            raise ValueError("At least one hyperedge is required")
        hyperedge_index = hyperedge_index.to(device=x.device, dtype=torch.long)
        if hyperedge_index.ndim != 2 or hyperedge_index.size(0) != 2:
            raise ValueError(
                "hyperedge_index must have shape [2, num_incident_pairs]"
            )
        if torch.any((hyperedge_type < 0) | (hyperedge_type >= self.NUM_REGION_TYPES)):
            raise ValueError("hyperedge_type values must be in [0, 2]")
        node_ids, hyperedge_ids = hyperedge_index
        if node_ids.numel() == 0:
            raise ValueError("Every hypergraph must contain node-hyperedge incidences")
        if torch.any((node_ids < 0) | (node_ids >= x.size(0))):
            raise ValueError("hyperedge_index contains an invalid node index")
        if torch.any((hyperedge_ids < 0) | (hyperedge_ids >= num_hyperedges)):
            raise ValueError("hyperedge_index contains an invalid hyperedge index")
        incidence_count = scatter(
            torch.ones_like(hyperedge_ids),
            hyperedge_ids,
            dim=0,
            dim_size=num_hyperedges,
            reduce="sum",
        )
        if torch.any(incidence_count == 0):
            raise ValueError("Empty hyperedges must be omitted from the variable layout")
        if batch is not None:
            edge_graph_min = scatter(batch[node_ids], hyperedge_ids, dim=0,
                                     dim_size=num_hyperedges, reduce='min')
            edge_graph_max = scatter(batch[node_ids], hyperedge_ids, dim=0,
                                     dim_size=num_hyperedges, reduce='max')
            if not torch.equal(edge_graph_min, edge_graph_max):
                raise ValueError("A hyperedge cannot contain nodes from multiple graphs")
        if self.is_v3:
            if torch.any(hyperedge_type.eq(self.FAR_FIELD)):
                raise ValueError("V3 omits far-field hyperedges; reconvert using the v3 schema")
            graph_ids = x.new_zeros(x.size(0), dtype=torch.long) if batch is None else batch
            edge_graph = scatter(graph_ids[node_ids], hyperedge_ids, dim=0,
                                 dim_size=num_hyperedges, reduce='min')
            core_count = scatter(hyperedge_type.eq(0).long(), edge_graph, dim=0,
                                 dim_size=num_graphs, reduce='sum')
            if torch.any(core_count.ne(1)):
                raise ValueError("V3 requires exactly one all-defect hyperedge per graph")
            inferred = infer_region_type(x.size(0), hyperedge_index, hyperedge_type)
            center_counts = scatter(inferred[node_ids].eq(0).long(), hyperedge_ids,
                                    dim=0, dim_size=num_hyperedges, reduce='sum')
            if torch.any(center_counts[hyperedge_type.eq(1)].ne(1)):
                raise ValueError("Each v3 local hyperedge requires exactly one defect center")

        if region_type is None:
            # Core incidences take priority over local and far incidences, so
            # an overlapping local center remains marked as a defect node.
            region_type = infer_region_type(x.size(0), hyperedge_index, hyperedge_type)
        else:
            region_type = region_type.to(device=x.device, dtype=torch.long).view(-1)
        if region_type.numel() != x.size(0):
            raise ValueError(
                f"Expected one region type per node ({x.size(0)}), "
                f"got {region_type.numel()}"
            )
        if torch.any((region_type < 0) | (region_type >= self.NUM_REGION_TYPES)):
            raise ValueError("region_type values must be in [0, 2] for every node")
        if self.is_v3 and not torch.equal(
                region_type, infer_region_type(x.size(0), hyperedge_index, hyperedge_type)):
            raise ValueError("region_type is inconsistent with v3 hyperedge membership")
        return (
            num_graphs,
            num_hyperedges,
            hyperedge_index,
            hyperedge_type,
            region_type,
        )

    def add_region_features(self, x, region_type):
        return x + self.region_embedding(region_type)

    def update(self, step, x, hyperedge_index, hyperedge_type, num_hyperedges):
        kwargs = {'hyperedge_type': hyperedge_type} if self.is_v3 else {}
        return self.blocks[step](
            x,
            hyperedge_index,
            self.region_embedding(hyperedge_type),
            num_hyperedges,
            **kwargs,
        )

    def pool(
            self,
            x,
            hyperedge_index,
            hyperedge_type,
            batch,
            num_graphs,
            num_hyperedges,
    ):
        """Apply the configured graph readout after checking batch isolation."""
        node_ids, hyperedge_ids = hyperedge_index
        incidence_graph = batch[node_ids]
        hyperedge_graph_min = scatter(
            incidence_graph,
            hyperedge_ids,
            dim=0,
            dim_size=num_hyperedges,
            reduce="min",
        )
        hyperedge_graph_max = scatter(
            incidence_graph,
            hyperedge_ids,
            dim=0,
            dim_size=num_hyperedges,
            reduce="max",
        )
        if not torch.equal(hyperedge_graph_min, hyperedge_graph_max):
            raise ValueError("A hyperedge cannot contain nodes from multiple graphs")
        if self.is_v3:
            return self.readout(x, hyperedge_index, hyperedge_type, batch,
                                num_graphs, num_hyperedges)

        hyperedge_pool = scatter(x[node_ids], hyperedge_ids, dim=0,
                                 dim_size=num_hyperedges, reduce='mean')
        type_pools = []
        for region_type in range(self.NUM_REGION_TYPES):
            mask = hyperedge_type.eq(region_type)
            if torch.any(mask):
                pooled = scatter(
                    hyperedge_pool[mask],
                    hyperedge_graph_min[mask],
                    dim=0,
                    dim_size=num_graphs,
                    reduce="mean",
                )
            else:
                pooled = x.new_zeros((num_graphs, self.hidden_dim))
            type_pools.append(pooled)
        return torch.cat(type_pools, dim=-1)


class RegionHypergraphNet(nn.Module):
    """Hypergraph model with independent per-defect neighborhoods.

    The variable hyperedges have three semantic types:

    0. one singleton core hyperedge for every defect;
    1. one local hyperedge per defect containing its center plus pristine atoms
       within the configured defect-neighbor radius;
    2. one optional far-field hyperedge containing remaining pristine atoms.

    Local hyperedges exclude other defects but may overlap on pristine atoms.
    V3 selects one all-defect edge, local edges, and hierarchical attention
    readout instead. Explicit schema defaults preserve direct legacy callers;
    new training configurations select V3.
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
            hypergraph_schema=LEGACY_HYPERGRAPH_SCHEMA,
            hypergraph_pooling=None,
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
            schema=hypergraph_schema,
            pooling=hypergraph_pooling,
        )
        if not self.hypergraph.defect_mean:
            self.state_embedding = nn.Sequential(
                nn.Linear(state_input_shape, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
        readout_dim = hidden_dim if self.hypergraph.defect_mean else 4 * hidden_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, 2 * hidden_dim),
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

        if state is None:
            state = x.new_zeros((num_graphs, self.state_embedding[0].in_features))
        else:
            state = state.to(device=x.device, dtype=x.dtype).reshape(num_graphs, -1)
        state_pool = self.state_embedding(state)
        return self.readout(torch.cat([region_pool, state_pool], dim=-1))
