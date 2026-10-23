"""On-device line graphs must retain legacy bond order, angles and gradients."""
from collections import defaultdict
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from HERA.models import alignn


def reference_angles(first, second, vectors, expansion):
    pairs = torch.tensor([first, second], device=vectors.device, dtype=torch.long)
    if not first:
        return pairs, vectors.new_empty((0, expansion.out_features))
    values = F.cosine_similarity(-vectors[pairs[0]], vectors[pairs[1]], dim=-1).clamp(-1., 1.)
    return pairs, expansion(values)


def reference_line_graph(indices, vectors, expansion):
    valid = (vectors.detach().norm(dim=-1) > 1e-8).cpu().tolist()
    sources, targets = indices.cpu().tolist()
    outgoing = defaultdict(list)
    for edge, source in enumerate(sources):
        if valid[edge]:
            outgoing[source].append(edge)
    first, second = [], []
    for edge, target in enumerate(targets):
        if valid[edge]:
            for other in outgoing[target]:
                if edge != other:
                    first.append(edge)
                    second.append(other)
    return reference_angles(first, second, vectors, expansion)


def reference_hetero_line_graph(indices, vectors, offsets, edge_types, expansion):
    valid = (vectors.detach().norm(dim=-1) > 1e-8).cpu().tolist()
    outgoing, targets = defaultdict(list), []
    for relation in edge_types:
        src_type, _, dst_type = relation
        source, target = indices[relation].cpu().tolist()
        for local, (src, dst) in enumerate(zip(source, target)):
            edge = offsets[relation] + local
            if valid[edge]:
                outgoing[src_type, src].append(edge)
                targets.append(((dst_type, dst), edge))
    first, second = [], []
    for target, edge in targets:
        for other in outgoing[target]:
            if edge != other:
                first.append(edge)
                second.append(other)
    return reference_angles(first, second, vectors, expansion)


class DeviceLineGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

    def assert_angles_and_gradients(self, actual, expected, vectors):
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
        if actual[0].numel():
            grad = torch.autograd.grad(actual[1].square().sum(), vectors, retain_graph=True)[0]
            reference = torch.autograd.grad(expected[1].square().sum(), vectors)[0]
            torch.testing.assert_close(grad, reference)

    def test_homogeneous_periodic_parallel_zero_and_disconnected_edges(self):
        for device in self.devices:
            for count in (0, 1, 7, 41):
                with self.subTest(device=device, edges=count):
                    torch.manual_seed(54 + count)
                    indices = torch.randint(0, 5, (2, count), device=device)
                    vectors = torch.randn(count, 3, device=device)
                    if count > 6:
                        indices[:, :4] = torch.tensor([[0, 0, 0, 1], [0, 1, 1, 0]], device=device)
                        vectors[4] = 0  # Invalid zero-length edge cannot form angles.
                        vectors[5] = 1e-10
                    vectors.requires_grad_()
                    expansion = alignn.AngleExpansion(8).to(device)
                    self.assert_angles_and_gradients(
                        alignn.build_line_graph(indices, vectors, expansion),
                        reference_line_graph(indices, vectors, expansion), vectors,
                    )
            indices = torch.tensor([[0, 0], [1, 2]], device=device)
            vectors = torch.ones(2, 3, device=device)
            pairs, angles = alignn.build_line_graph(indices, vectors, alignn.AngleExpansion(8).to(device))
            self.assertEqual(tuple(pairs.shape), (2, 0))
            self.assertEqual(tuple(angles.shape), (0, 8))
            pairs, _ = alignn.build_line_graph(indices, torch.zeros_like(vectors), alignn.AngleExpansion(8).to(device))
            self.assertEqual(tuple(pairs.shape), (2, 0))

    def test_heterogeneous_namespaces_empty_relations_and_no_dd(self):
        relations = [('atom', 'aa', 'atom'), ('defect', 'dd', 'defect'),
                     ('atom', 'ad', 'defect'), ('defect', 'da', 'atom')]
        for device in self.devices:
            for drop_dd in (False, True):
                torch.manual_seed(7)
                types = [r for r in relations if not drop_dd or r[1] != 'dd']
                indices, offsets, parts, size = {}, {}, [], 0
                for relation in types:
                    count = 0 if relation[1] == 'dd' else 11
                    indices[relation] = torch.randint(0, 3, (2, count), device=device)
                    offsets[relation] = size
                    size += count
                    parts.append(torch.randn(count, 3, device=device))
                vectors = torch.cat(parts)
                vectors[4] = 0
                vectors.requires_grad_()
                expansion = alignn.AngleExpansion(8).to(device)
                self.assert_angles_and_gradients(
                    alignn.build_hetero_line_graph(indices, vectors, offsets, types, expansion),
                    reference_hetero_line_graph(indices, vectors, offsets, types, expansion), vectors,
                )

    def test_model_predictions_parameter_gradients_and_batchnorm_buffers(self):
        for device in self.devices:
            torch.manual_seed(32)
            model = alignn.ALIGNN(4, 6, hidden_dim=8, n_blocks=1, gcn_blocks=1).to(device)
            original = deepcopy(model)
            indices = torch.tensor([[0, 1, 1, 2, 2, 0, 3, 4, 4, 3],
                                    [1, 0, 2, 1, 0, 2, 4, 3, 4, 4]], device=device)
            inputs = (torch.randn(5, 4, device=device), indices,
                      torch.randn(10, 6, device=device), torch.tensor([0, 0, 0, 1, 1], device=device))
            vectors = torch.randn(10, 3, device=device)
            output = model(*inputs, edge_vec=vectors)
            output.square().sum().backward()
            with patch.object(alignn, 'build_line_graph', reference_line_graph):
                expected = original(*inputs, edge_vec=vectors)
                expected.square().sum().backward()
            torch.testing.assert_close(output, expected)
            for actual, reference in zip(model.parameters(), original.parameters()):
                if actual.grad is not None:
                    torch.testing.assert_close(actual.grad, reference.grad)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, original.state_dict()[name])

    def test_hetero_model_predictions_and_parameter_gradients(self):
        for device in self.devices:
            for drop_dd in (False, True):
                torch.manual_seed(32)
                relations = [('atom', 'aa', 'atom'), ('atom', 'ad', 'defect'),
                             ('defect', 'da', 'atom')]
                if not drop_dd:
                    relations.append(('defect', 'dd', 'defect'))
                model = alignn.HeteroALIGNN(
                    4, 6, (['atom', 'defect'], relations), hidden_dim=8,
                    n_blocks=1, gcn_blocks=1, relation_mode='shared_residual',
                    feature_norm='layernorm', pooling='defect_energy_mean',
                ).to(device)
                original = deepcopy(model)
                x = {node: torch.randn(3, 4, device=device) for node in ('atom', 'defect')}
                indices = {rel: torch.randint(0, 3, (2, 8), device=device) for rel in relations}
                attrs = {rel: torch.randn(8, 6, device=device) for rel in relations}
                vectors = {rel: torch.randn(8, 3, device=device) for rel in relations}
                batch = {node: torch.zeros(3, dtype=torch.long, device=device) for node in x}
                output = model(x, indices, attrs, batch, edge_vec_dict=vectors)
                output.square().sum().backward()
                with patch.object(alignn, 'build_hetero_line_graph', reference_hetero_line_graph):
                    expected = original(x, indices, attrs, batch, edge_vec_dict=vectors)
                    expected.square().sum().backward()
                torch.testing.assert_close(output, expected)
                for actual, reference in zip(model.parameters(), original.parameters()):
                    if actual.grad is not None:
                        torch.testing.assert_close(actual.grad, reference.grad)


if __name__ == '__main__':
    unittest.main()
