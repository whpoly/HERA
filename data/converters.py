"""Crystal structure → PyG Data converters and feature extractors."""

import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies

from ..utils.scaler import MyTensor


# ------------------------------------------------------------------ #
#  Distance converters
# ------------------------------------------------------------------ #

class DummyConverter:
    def convert(self, d):
        return d.reshape((-1, 1))


class GaussianDistanceConverter:
    def __init__(self, centers=np.linspace(0, 5, 100), sigma=0.5):
        self.centers = centers
        self.sigma = sigma

    def convert(self, d):
        return np.exp(
            -((d.reshape((-1, 1)) - self.centers.reshape((1, -1))) / self.sigma) ** 2
        )

    def get_shape(self, eos=False):
        shape = len(self.centers)
        if eos:
            shape += 2
        return shape


class FlattenGaussianDistanceConverter(GaussianDistanceConverter):
    def __init__(self, centers=np.linspace(0, 5, 100), sigma=0.5):
        super().__init__(centers, sigma)

    def convert(self, d):
        res = []
        for arr in d:
            res.append(super().convert(arr))
        return np.hstack(res)

    def get_shape(self, eos=False):
        shape = 2 * len(self.centers)
        if eos:
            shape += 2
        return shape


# ------------------------------------------------------------------ #
#  Atom feature extractor
# ------------------------------------------------------------------ #

class AtomFeaturesExtractor:
    def __init__(self, atom_features, task):
        self.atom_features = atom_features
        self.task = task.split('_')[0]

    def convert(self, structure: Structure):
        # elem_embedding must be set globally before use (loaded in main.py)
        from ..data.datasets import elem_embedding

        if self.atom_features == "Z":
            if self.task == 'megnet':
                return np.array(
                    [0 if isinstance(i, DummySpecies) else i.Z for i in structure.species]
                ).reshape(-1, 1)
            else:
                return np.array([
                    [0] * 92 if isinstance(i.specie, DummySpecies)
                    else elem_embedding[i.specie.Z]
                    for i in structure.sites
                ])
        elif self.atom_features == 'werespecies':
            return np.array([
                [
                    0 if isinstance(i.specie, DummySpecies) else i.specie.Z,
                    i.properties["was"],
                ] for i in structure.sites
            ])
        else:
            raise NotImplementedError

    def get_shape(self):
        if self.atom_features == "Z":
            if self.task == 'megnet':
                return None
            else:
                return 92
        elif self.atom_features == 'werespecies':
            return 2
        else:
            return None


# ------------------------------------------------------------------ #
#  Structure → PyG Data converter
# ------------------------------------------------------------------ #

class SimpleCrystalConverter:
    def __init__(
            self,
            task,
            atom_converter=None,
            bond_converter=None,
            add_z_bond_coord=False,
            add_eos_features=False,
            cutoff=5.0,
            ignore_state=False,
    ):
        self.cutoff = cutoff
        self.atom_converter = atom_converter if atom_converter else DummyConverter()
        self.bond_converter = bond_converter if bond_converter else DummyConverter()
        self.add_z_bond_coord = add_z_bond_coord
        self.add_eos_features = add_eos_features
        self.ignore_state = ignore_state
        task_suffix = task.split('_')[1]
        self.task = task_suffix

    def convert(self, d):
        if self.task in ('sparse', 'full'):
            bond_index = [[], []]
            bond_attr = []
            all_nbrs = d.get_all_neighbors(self.cutoff, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            for i, nbrs in enumerate(all_nbrs):
                bond_index[0] += [i] * len(nbrs)
                bond_index[1].extend(list(map(lambda x: x[2], nbrs)))
                bond_attr.extend(list(map(lambda x: x[1], nbrs)))
            edge_index = torch.LongTensor(np.array(bond_index))

            x = torch.Tensor(self.atom_converter.convert(d))
            distances_preprocessed = np.array(bond_attr)
            if self.add_z_bond_coord:
                cart_coords = np.array(d.cart_coords)
                z_coord_diff = np.abs(cart_coords[edge_index[0], 2] - cart_coords[edge_index[1], 2])
                distances_preprocessed = np.stack(
                    (distances_preprocessed, z_coord_diff), axis=0
                )

            edge_attr = torch.Tensor(self.bond_converter.convert(distances_preprocessed))

            if self.ignore_state:
                state = [[0.0, 0.0]]
            else:
                state = getattr(d, "state", None) or [[0.0, 0.0]]
            if len(state[0]) > 2:
                raise NotImplementedError("We currently only support state length of 1 and 2")
            if len(state[0]) == 1:
                state[0].append(state[0][0])
            y = d.y if hasattr(d, "y") else 0
            weight = d.weight if hasattr(d, 'weight') else 1
            bond_batch = MyTensor(np.zeros(edge_index.shape[1])).long()

            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                state=torch.Tensor(state),
                y=y,
                bond_batch=bond_batch,
                weight=weight,
                structure=d,
            )
        elif self.task == 'attention':
            # Homogeneous graph with explicit node_type (0=host, 1=defect)
            node_type = torch.LongTensor([int(site.properties['type']) for site in d])

            bond_index = [[], []]
            bond_attr = []
            all_nbrs = d.get_all_neighbors(self.cutoff, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            for i, nbrs in enumerate(all_nbrs):
                bond_index[0] += [i] * len(nbrs)
                bond_index[1].extend(list(map(lambda x: x[2], nbrs)))
                bond_attr.extend(list(map(lambda x: x[1], nbrs)))
            edge_index = torch.LongTensor(np.array(bond_index))

            x = torch.Tensor(self.atom_converter.convert(d))
            distances_preprocessed = np.array(bond_attr)
            if self.add_z_bond_coord:
                cart_coords = np.array(d.cart_coords)
                z_coord_diff = np.abs(cart_coords[edge_index[0], 2] - cart_coords[edge_index[1], 2])
                distances_preprocessed = np.stack(
                    (distances_preprocessed, z_coord_diff), axis=0
                )

            edge_attr = torch.Tensor(self.bond_converter.convert(distances_preprocessed))

            if self.ignore_state:
                state = [[0.0, 0.0]]
            else:
                state = getattr(d, "state", None) or [[0.0, 0.0]]
            if len(state[0]) > 2:
                raise NotImplementedError("We currently only support state length of 1 and 2")
            if len(state[0]) == 1:
                state[0].append(state[0][0])
            y = d.y if hasattr(d, "y") else 0
            weight = d.weight if hasattr(d, 'weight') else 1
            bond_batch = MyTensor(np.zeros(edge_index.shape[1])).long()

            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                state=torch.Tensor(state),
                y=y,
                bond_batch=bond_batch,
                weight=weight,
                node_type=node_type,
                structure=d,
            )
        else:
            # hetero / local mode
            bond_index = [[], []]
            bond_attr = []
            indexs = torch.LongTensor([site.properties['type'] for site in d])
            all_nbrs = d.get_all_neighbors(self.cutoff, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            for i, nbrs in enumerate(all_nbrs):
                bond_index[0] += [i]
                bond_index[1].extend([i])
                bond_attr.extend([0.0])
                for j in nbrs:
                    bond_index[0] += [i]
                    bond_index[1].extend([j[2]])
                    bond_attr.extend([j[1]])

            edge_index = torch.LongTensor(np.array(bond_index))
            x = torch.Tensor(self.atom_converter.convert(d))
            distances_preprocessed = np.array(bond_attr)
            edge_attr = torch.Tensor(self.bond_converter.convert(distances_preprocessed))

            if self.ignore_state:
                state = [[0.0, 0.0]]
            else:
                state = getattr(d, "state", None) or [[0.0, 0.0]]
            if len(state[0]) > 2:
                raise NotImplementedError("We currently only support state length of 1 and 2")
            if len(state[0]) == 1:
                state[0].append(state[0][0])
            y = d.y if hasattr(d, "y") else 0
            weight = d.weight if hasattr(d, 'weight') else 1
            bond_batch = MyTensor(np.zeros(edge_index.shape[1])).long()
            edge_incidies = torch.LongTensor(np.zeros(edge_index.shape[1]))
            for i in range(edge_index.shape[1]):
                if indexs[edge_index[0][i]] == 0 and indexs[edge_index[1][i]] == 0:
                    edge_incidies[i] = 0
                elif indexs[edge_index[0][i]] == 1 and indexs[edge_index[1][i]] == 1:
                    edge_incidies[i] = 1
                elif indexs[edge_index[0][i]] == 0 and indexs[edge_index[1][i]] == 1:
                    edge_incidies[i] = 2
                else:
                    edge_incidies[i] = 3

            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                state=torch.Tensor(state),
                y=y,
                bond_batch=bond_batch,
                weight=weight,
                structure=(d, indexs),
            ).to_heterogeneous(
                node_type=indexs,
                edge_type=edge_incidies,
                node_type_names=['atom', 'defect'],
                edge_type_names=[
                    ('atom', 'aa', 'atom'),
                    ('defect', 'dd', 'defect'),
                    ('atom', 'ad', 'defect'),
                    ('defect', 'da', 'atom'),
                ],
            )

    def __call__(self, d):
        return self.convert(d)
