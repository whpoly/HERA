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
        self.task = self._task_family(task)

    @staticmethod
    def _task_family(task):
        if task.startswith('megnet_'):
            return 'megnet'
        if task.startswith('cgcnn_') or task == 'hetero_cgcnn_was':
            return 'cgcnn'
        if task.startswith('definet_'):
            return 'definet'
        return task.split('_')[0]

    @staticmethod
    def _zero_embedding(elem_embedding):
        if elem_embedding:
            return [0] * len(next(iter(elem_embedding.values())))
        return [0] * 92

    @classmethod
    def _embedding_from_z(cls, z, elem_embedding):
        if z is None or int(z) <= 0:
            return cls._zero_embedding(elem_embedding)
        return elem_embedding[int(z)]

    @staticmethod
    def _current_z(site):
        if isinstance(site.specie, DummySpecies):
            return 0
        return site.specie.Z

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
        elif self.atom_features == 'was_species':
            features = []
            for site in structure.sites:
                current_z = self._current_z(site)
                previous_z = site.properties.get('was', current_z)
                if self.task == 'megnet':
                    features.append([float(current_z), float(previous_z)])
                else:
                    features.append(
                        self._embedding_from_z(current_z, elem_embedding)
                        + self._embedding_from_z(previous_z, elem_embedding)
                    )
            return np.array(features)
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
        elif self.atom_features == 'was_species':
            if self.task == 'megnet':
                return 2
            return 184
        else:
            return None


# ------------------------------------------------------------------ #
#  Structure → PyG Data converter
# ------------------------------------------------------------------ #

class SimpleCrystalConverter:
    NODE_TYPE_NAMES = ['atom', 'defect']
    EDGE_TYPE_NAMES = [
        ('atom', 'aa', 'atom'),
        ('defect', 'dd', 'defect'),
        ('atom', 'ad', 'defect'),
        ('defect', 'da', 'atom'),
    ]
    LOCAL_STRUCTURE_MODES = (
        'local',
        'hetero_local',
        'hetero_local_was',
        'attention_local',
        'attention_local_was',
    )
    ATTENTION_MODES = ('attention', 'attention_local', 'attention_was', 'attention_local_was')

    def __init__(
            self,
            task,
            atom_converter=None,
            bond_converter=None,
            add_z_bond_coord=False,
            add_eos_features=False,
            cutoff=6.0,
            local_radius=None,
            ignore_state=False,
    ):
        self.cutoff = cutoff
        self.local_radius = cutoff if local_radius is None else local_radius
        self.atom_converter = atom_converter if atom_converter else DummyConverter()
        self.bond_converter = bond_converter if bond_converter else DummyConverter()
        self.add_z_bond_coord = add_z_bond_coord
        self.add_eos_features = add_eos_features
        self.ignore_state = ignore_state
        self.task = self._graph_mode(task)

    @staticmethod
    def _graph_mode(task):
        if task == 'hetero_cgcnn_was':
            return 'hetero_was'
        parts = task.split('_', 1)
        return parts[1] if len(parts) > 1 else task

    @staticmethod
    def _copy_structure_metadata(source, target):
        for attr in ("source_id", "source_name", "source_path", "state", "y", "weight"):
            if hasattr(source, attr):
                setattr(target, attr, getattr(source, attr))
        return target

    @staticmethod
    def _source_metadata_kwargs(structure):
        return {
            "source_id": getattr(structure, "source_id", ""),
            "source_name": getattr(structure, "source_name", ""),
            "source_path": getattr(structure, "source_path", ""),
        }

    def _local_radius_structure(self, structure):
        defect_indices = [
            idx for idx, site in enumerate(structure)
            if int(site.properties.get('type', 0)) == 1
        ]
        if not defect_indices:
            return self._copy_structure_metadata(structure, structure.copy())

        selected = set(defect_indices)
        for idx, site in enumerate(structure):
            if idx in selected:
                continue
            # Union of all defect-centered radius neighborhoods.
            min_distance = min(structure.get_distance(idx, defect_idx) for defect_idx in defect_indices)
            if self.local_radius > 0 and min_distance <= self.local_radius:
                selected.add(idx)

        local_structure = Structure.from_sites([structure[idx] for idx in sorted(selected)])
        return self._copy_structure_metadata(structure, local_structure)

    def convert(self, d):
        if self.task == 'sparse':
            raise ValueError("Sparse mode has been removed; use local mode with --r 0 instead.")

        if self.task in self.LOCAL_STRUCTURE_MODES:
            d = self._local_radius_structure(d)

        if self.task in ('full', 'was', 'local'):
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
                **self._source_metadata_kwargs(d),
            )
        elif self.task in self.ATTENTION_MODES:
            # Homogeneous graph with explicit node_type (0=host, 1=defect).
            # defect_marker uses the same binary split: 0=pristine, 1=defect.
            node_type = torch.LongTensor([int(site.properties['type']) for site in d])
            defect_marker = torch.LongTensor([
                0 if int(site.properties['type']) == 0
                else 1
                for site in d
            ])

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
                defect_marker=defect_marker,
                structure=d,
                **self._source_metadata_kwargs(d),
            )
        else:
            # hetero mode
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

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                weight=weight,
                structure=(d, indexs),
                **self._source_metadata_kwargs(d),
            ).to_heterogeneous(
                node_type=indexs,
                edge_type=edge_incidies,
                node_type_names=self.NODE_TYPE_NAMES,
                edge_type_names=self.EDGE_TYPE_NAMES,
            )
            data.state = torch.Tensor(state)
            for edge_type_idx, edge_type_name in enumerate(self.EDGE_TYPE_NAMES):
                edge_count = int((edge_incidies == edge_type_idx).sum().item())
                data[edge_type_name].bond_batch = MyTensor(np.zeros(edge_count)).long()
            return data

    def __call__(self, d):
        return self.convert(d)
