"""Trainer class that handles model construction, data preparation, training, and evaluation."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch_geometric.nn import to_hetero

from ..data.converters import (
    SimpleCrystalConverter,
    GaussianDistanceConverter,
    FlattenGaussianDistanceConverter,
    AtomFeaturesExtractor,
)
from ..models.megnet import MEGNet, HeteroMEGNet, AttentionMEGNet
from ..models.cgcnn import CGCNN, CrystalGraphConvNet, Heterocgcnn, AttentionCGCNN, DefiNet
from ..models.alignn import ALIGNN, HeteroALIGNN, AttentionALIGNN, DefiNetALIGNN
from ..utils.scaler import Scaler
from .losses import MAELoss


CGCNN_ATTENTION_TASKS = (
    'cgcnn_attention',
    'cgcnn_attention_local',
    'cgcnn_attention_was',
    'cgcnn_attention_local_was',
)
CGCNN_HETERO_TASKS = (
    'cgcnn_hetero',
    'cgcnn_hetero_fixed_pool',
    'hetero_cgcnn_was',
    'cgcnn_hetero_local',
    'cgcnn_hetero_local_was',
)
MEGNET_HETERO_TASKS = (
    'megnet_hetero',
    'megnet_hetero_fixed_pool',
    'megnet_hetero_was',
    'megnet_hetero_local',
    'megnet_hetero_local_was',
)
MEGNET_ATTENTION_TASKS = (
    'megnet_attention',
    'megnet_attention_local',
    'megnet_attention_was',
    'megnet_attention_local_was',
)
ALIGNN_HOMOGENEOUS_TASKS = (
    'alignn_full',
    'alignn_full_x',
    'alignn_local',
    'alignn_was_x',
)
ALIGNN_HETERO_TASKS = (
    'alignn_hetero',
    'alignn_hetero_fixed_pool',
    'alignn_hetero_was',
    'alignn_hetero_local',
    'alignn_hetero_local_was',
)
ALIGNN_ATTENTION_TASKS = (
    'alignn_attention',
    'alignn_attention_local',
    'alignn_attention_was',
    'alignn_attention_local_was',
)
ALIGNN_DEFINET_TASKS = (
    'alignn_definet',
    'alignn_definet_local',
    'alignn_definet_was',
    'alignn_definet_local_was',
)
DEFINET_ATTENTION_TASKS = (
    'definet_attention',
    'definet_attention_local',
    'definet_attention_was',
    'definet_attention_local_was',
)
HETERO_NODE_TYPES = ('atom', 'defect')
HETERO_EDGE_TYPES = (
    ('atom', 'aa', 'atom'),
    ('defect', 'dd', 'defect'),
    ('atom', 'ad', 'defect'),
    ('defect', 'da', 'atom'),
)


def set_attr(structure, attr, name):
    setattr(structure, name, attr)
    return structure


def _complete_hetero_inputs(x_dict, edge_index_dict, edge_attr_dict, batch_dict, bond_batch_dict=None):
    x_dict = dict(x_dict)
    edge_index_dict = dict(edge_index_dict)
    edge_attr_dict = dict(edge_attr_dict)
    batch_dict = dict(batch_dict)
    bond_batch_dict = None if bond_batch_dict is None else dict(bond_batch_dict)

    ref_x = next(iter(x_dict.values()))
    node_feature_dim = ref_x.shape[1] if ref_x.dim() > 1 else 1
    for node_type in HETERO_NODE_TYPES:
        if node_type not in x_dict:
            x_dict[node_type] = ref_x.new_empty((0, node_feature_dim))
        if node_type not in batch_dict:
            batch_dict[node_type] = torch.empty((0,), dtype=torch.long, device=ref_x.device)

    if edge_attr_dict:
        ref_edge_attr = next(iter(edge_attr_dict.values()))
        edge_feature_dim = ref_edge_attr.shape[1] if ref_edge_attr.dim() > 1 else 1
    else:
        ref_edge_attr = ref_x.new_empty((0, 1))
        edge_feature_dim = 1

    for edge_type in HETERO_EDGE_TYPES:
        if edge_type not in edge_index_dict:
            edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=ref_x.device)
        if edge_type not in edge_attr_dict:
            edge_attr_dict[edge_type] = ref_edge_attr.new_empty((0, edge_feature_dim))
        if bond_batch_dict is not None and edge_type not in bond_batch_dict:
            bond_batch_dict[edge_type] = torch.empty((0,), dtype=torch.long, device=ref_x.device)

    return x_dict, edge_index_dict, edge_attr_dict, batch_dict, bond_batch_dict


def _complete_hetero_edge_vecs(edge_vec_dict, edge_attr_dict):
    edge_vec_dict = {} if edge_vec_dict is None else dict(edge_vec_dict)
    for edge_type in HETERO_EDGE_TYPES:
        if edge_type not in edge_vec_dict:
            edge_vec_dict[edge_type] = edge_attr_dict[edge_type].new_zeros(
                (edge_attr_dict[edge_type].size(0), 3)
            )
    return edge_vec_dict


def _collect_hetero_attr(batch, attr):
    try:
        return batch.collect(attr)
    except (AttributeError, KeyError):
        return getattr(batch, f'{attr}_dict', None)


def _prediction_vector(preds):
    return preds.reshape(-1)


class MEGNetTrainer:
    def __init__(self, config, device, seed=None):
        self.config = config
        self.device = device
        self.seed = None if seed is None else int(seed)

        if self.config["model"]["add_z_bond_coord"]:
            bond_converter = FlattenGaussianDistanceConverter(
                centers=np.linspace(0, self.config['model']['cutoff'], self.config['model']['edge_embed_size'])
            )
        else:
            bond_converter = GaussianDistanceConverter(
                centers=np.linspace(0, self.config['model']['cutoff'], self.config['model']['edge_embed_size'])
            )
        atom_converter = AtomFeaturesExtractor(self.config["model"]["atom_features"], self.config['task'])
        self.converter = SimpleCrystalConverter(
            self.config['task'],
            bond_converter=bond_converter,
            atom_converter=atom_converter,
            cutoff=self.config["model"]["cutoff"],
            local_radius=self.config["model"].get("local_radius", self.config["model"]["cutoff"]),
            add_z_bond_coord=self.config["model"]["add_z_bond_coord"],
            add_eos_features=(use_eos := self.config["model"].get("add_eos_features", False)),
        )
        self.scaler = Scaler()

        task = self.config['task']
        # Build model based on task string:  {model}_{mode}
        if task in ('megnet_full', 'megnet_full_x', 'megnet_local', 'megnet_was_x'):
            self.model = MEGNet(
                edge_input_shape=bond_converter.get_shape(eos=use_eos),
                node_input_shape=atom_converter.get_shape(),
                embedding_size=self.config['model']['embedding_size'],
                n_blocks=self.config['model']['nblocks'],
                state_input_shape=self.config["model"]["state_input_shape"],
                vertex_aggregation=self.config["model"]["vertex_aggregation"],
                global_aggregation=self.config["model"]["global_aggregation"],
            ).to(self.device)
        elif task in MEGNET_HETERO_TASKS:
            self.model = HeteroMEGNet(
                edge_input_shape=bond_converter.get_shape(eos=use_eos),
                node_input_shape=atom_converter.get_shape(),
                embedding_size=self.config['model']['embedding_size'],
                metadata=(['atom', 'defect'],
                          [('atom', 'aa', 'atom'),
                           ('defect', 'dd', 'defect'),
                           ('atom', 'ad', 'defect'),
                           ('defect', 'da', 'atom')]),
                n_blocks=self.config['model']['nblocks'],
                state_input_shape=self.config["model"]["state_input_shape"],
                vertex_aggregation=self.config["model"]["vertex_aggregation"],
                global_aggregation=self.config["model"]["global_aggregation"],
                fixed_pooling=self.config["model"].get("fixed_pooling", False),
            ).to(self.device)
        elif task in CGCNN_HETERO_TASKS:
            model = CrystalGraphConvNet(
                orig_atom_fea_len=atom_converter.get_shape(),
                nbr_fea_len=bond_converter.get_shape(eos=use_eos),
                n_conv=self.config['model']['nblocks'],
            )
            model = to_hetero(model, metadata=(['atom', 'defect'],
                                                [('atom', 'aa', 'atom'),
                                                 ('defect', 'dd', 'defect'),
                                                 ('atom', 'ad', 'defect'),
                                                 ('defect', 'da', 'atom')]), aggr='mean')
            self.model = Heterocgcnn(
                model,
                orig_atom_fea_len=atom_converter.get_shape(),
                nbr_fea_len=bond_converter.get_shape(eos=use_eos),
                n_h=3,
                fixed_pooling=self.config["model"].get("fixed_pooling", False),
            ).to(self.device)
        elif task in ALIGNN_HOMOGENEOUS_TASKS:
            self.model = ALIGNN(
                node_input_shape=atom_converter.get_shape(),
                edge_input_shape=bond_converter.get_shape(eos=use_eos),
                hidden_dim=self.config['model']['embedding_size'],
                n_blocks=self.config['model']['nblocks'],
                angle_embed_size=self.config['model'].get(
                    'angle_embed_size',
                    self.config['model']['edge_embed_size'],
                ),
                vertex_aggregation=self.config["model"]["vertex_aggregation"],
            ).to(self.device)
        elif task in ALIGNN_HETERO_TASKS:
            self.model = HeteroALIGNN(
                node_input_shape=atom_converter.get_shape(),
                edge_input_shape=bond_converter.get_shape(eos=use_eos),
                metadata=(['atom', 'defect'],
                          [('atom', 'aa', 'atom'),
                           ('defect', 'dd', 'defect'),
                           ('atom', 'ad', 'defect'),
                           ('defect', 'da', 'atom')]),
                hidden_dim=self.config['model']['embedding_size'],
                n_blocks=self.config['model']['nblocks'],
                angle_embed_size=self.config['model'].get(
                    'angle_embed_size',
                    self.config['model']['edge_embed_size'],
                ),
                vertex_aggregation=self.config["model"]["vertex_aggregation"],
                fixed_pooling=self.config["model"].get("fixed_pooling", False),
            ).to(self.device)
        elif task in ALIGNN_ATTENTION_TASKS:
            self.model = AttentionALIGNN(
                node_input_shape=atom_converter.get_shape(),
                edge_input_shape=bond_converter.get_shape(eos=use_eos),
                hidden_dim=self.config['model']['embedding_size'],
                n_blocks=self.config['model']['nblocks'],
                angle_embed_size=self.config['model'].get(
                    'angle_embed_size',
                    self.config['model']['edge_embed_size'],
                ),
                n_heads=self.config['model'].get('n_heads', 4),
                vertex_aggregation=self.config["model"]["vertex_aggregation"],
            ).to(self.device)
        elif task in ALIGNN_DEFINET_TASKS:
            self.model = DefiNetALIGNN(
                node_input_shape=atom_converter.get_shape(),
                edge_input_shape=bond_converter.get_shape(eos=use_eos),
                hidden_dim=self.config['model']['embedding_size'],
                n_blocks=self.config['model']['nblocks'],
                angle_embed_size=self.config['model'].get(
                    'angle_embed_size',
                    self.config['model']['edge_embed_size'],
                ),
                n_marker_types=self.config['model'].get('n_marker_types', 2),
                vertex_aggregation=self.config["model"]["vertex_aggregation"],
            ).to(self.device)
        elif task in MEGNET_ATTENTION_TASKS:
            self.model = AttentionMEGNet(
                edge_input_shape=bond_converter.get_shape(eos=use_eos),
                node_input_shape=atom_converter.get_shape(),
                embedding_size=self.config['model']['embedding_size'],
                n_blocks=self.config['model']['nblocks'],
                n_heads=self.config['model'].get('n_heads', 4),
                state_input_shape=self.config["model"]["state_input_shape"],
                vertex_aggregation=self.config["model"]["vertex_aggregation"],
                global_aggregation=self.config["model"]["global_aggregation"],
            ).to(self.device)
        elif task in CGCNN_ATTENTION_TASKS:
            self.model = AttentionCGCNN(
                orig_atom_fea_len=atom_converter.get_shape(),
                nbr_fea_len=bond_converter.get_shape(eos=use_eos),
                n_conv=self.config['model']['nblocks'],
                n_heads=self.config['model'].get('n_heads', 4),
                n_h=3,
            ).to(self.device)
        elif task in DEFINET_ATTENTION_TASKS:
            self.model = DefiNet(
                orig_atom_fea_len=atom_converter.get_shape(),
                nbr_fea_len=bond_converter.get_shape(eos=use_eos),
                atom_fea_len=self.config['model']['embedding_size'],
                n_conv=self.config['model']['nblocks'],
                n_marker_types=self.config['model'].get('n_marker_types', 2),
                n_h=3,
            ).to(self.device)
        else:
            # default: homogeneous CGCNN variants
            self.model = CGCNN(
                orig_atom_fea_len=atom_converter.get_shape(),
                nbr_fea_len=bond_converter.get_shape(eos=use_eos),
                n_conv=self.config['model']['nblocks'],
                n_h=3,
            ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["optim"]["lr_initial"],
        )

        if self.config["optim"]["scheduler"].lower() == "reducelronplateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.config["optim"]["factor"],
                patience=self.config["optim"]["patience"],
                threshold=self.config["optim"]["threshold"],
                min_lr=self.config["optim"]["min_lr"],
            )
        else:
            raise ValueError("Unknown scheduler")

    def _make_generator(self, offset=0):
        if self.seed is None:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(offset))
        return generator

    # -------------------------------------------------------------- #
    #  Data preparation
    # -------------------------------------------------------------- #

    def prepare_data(self, train_data, train_targets, test_data, test_targets,
                     target_name, train_weights=None, test_weights=None):
        print('adding targets to data')
        train_data = [set_attr(s, y, 'y') for s, y in zip(train_data, train_targets)]
        test_data = [set_attr(s, y, 'y') for s, y in zip(test_data, test_targets)]
        if test_weights is not None:
            test_data = [set_attr(s, w, 'weight') for s, w in zip(test_data, test_weights)]

        print("converting data")
        self.train_structures = [self.converter.convert(s) for s in tqdm(train_data)]
        self.test_structures = [self.converter.convert(s) for s in tqdm(test_data)]
        self.scaler.fit(self.train_structures)

        self.sampler = None
        if train_weights is not None:
            self.sampler = WeightedRandomSampler(
                torch.tensor(train_weights).float(),
                len(train_weights),
                generator=self._make_generator(1),
            )

        self.trainloader = DataLoader(
            self.train_structures,
            batch_size=self.config["model"]["train_batch_size"],
            shuffle=False if train_weights is not None else True,
            num_workers=0,
            sampler=self.sampler,
            generator=self._make_generator(2),
        )
        self.testloader = DataLoader(
            self.test_structures,
            batch_size=self.config["model"]["test_batch_size"],
            shuffle=False,
            num_workers=0,
            generator=self._make_generator(3),
        )
        self.target_name = target_name

    # -------------------------------------------------------------- #
    #  Forward helper (dispatches by task)
    # -------------------------------------------------------------- #

    def _forward(self, batch):
        task = self.config['task']
        if task in MEGNET_HETERO_TASKS:
            x_dict, edge_index_dict, edge_attr_dict, batch_dict, bond_batch_dict = _complete_hetero_inputs(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict,
                batch.batch_dict,
                batch.bond_batch_dict,
            )
            return _prediction_vector(self.model(
                x_dict, edge_index_dict, edge_attr_dict,
                batch.state, batch_dict, bond_batch_dict,
                pool_type=_collect_hetero_attr(batch, 'pool_type'),
            ))
        elif task in CGCNN_HETERO_TASKS:
            x_dict, edge_index_dict, edge_attr_dict, batch_dict, _ = _complete_hetero_inputs(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict,
                batch.batch_dict,
            )
            return _prediction_vector(self.model(
                x_dict, edge_index_dict, edge_attr_dict,
                batch_dict,
                pool_type=_collect_hetero_attr(batch, 'pool_type'),
            ))
        elif task in ALIGNN_HETERO_TASKS:
            x_dict, edge_index_dict, edge_attr_dict, batch_dict, _ = _complete_hetero_inputs(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict,
                batch.batch_dict,
            )
            edge_vec_dict = _complete_hetero_edge_vecs(
                _collect_hetero_attr(batch, 'edge_vec'),
                edge_attr_dict,
            )
            return _prediction_vector(self.model(
                x_dict, edge_index_dict, edge_attr_dict,
                batch_dict,
                edge_vec_dict=edge_vec_dict,
                state=batch.state,
                pool_type=_collect_hetero_attr(batch, 'pool_type'),
            ))
        elif task in ALIGNN_HOMOGENEOUS_TASKS:
            return _prediction_vector(self.model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                edge_vec=getattr(batch, 'edge_vec', None),
            ))
        elif task in ALIGNN_ATTENTION_TASKS:
            return _prediction_vector(self.model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                edge_vec=getattr(batch, 'edge_vec', None),
                node_type=batch.node_type,
            ))
        elif task in ALIGNN_DEFINET_TASKS:
            marker = getattr(batch, 'defect_marker', getattr(batch, 'node_type', None))
            return _prediction_vector(self.model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                edge_vec=getattr(batch, 'edge_vec', None),
                defect_marker=marker,
            ))
        elif task in ('cgcnn_full', 'cgcnn_full_x', 'cgcnn_local', 'cgcnn_was_x'):
            return _prediction_vector(self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
        elif task in CGCNN_ATTENTION_TASKS:
            return _prediction_vector(self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, node_type=batch.node_type))
        elif task in DEFINET_ATTENTION_TASKS:
            marker = getattr(batch, 'defect_marker', getattr(batch, 'node_type', None))
            return _prediction_vector(self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, defect_marker=marker))
        elif task in MEGNET_ATTENTION_TASKS:
            return _prediction_vector(self.model(batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch, node_type=batch.node_type))
        else:
            return _prediction_vector(self.model(batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch))

    # -------------------------------------------------------------- #
    #  Training / evaluation
    # -------------------------------------------------------------- #

    def train_one_epoch(self):
        mses, maes = [], []
        self.model.train(True)
        for batch in self.trainloader:
            self.optimizer.zero_grad(set_to_none=True)
            batch = batch.to(self.device)
            preds = self._forward(batch)
            loss = F.mse_loss(self.scaler.transform(batch.y), preds, reduction='mean')
            loss.backward()
            self.optimizer.step()
            mses.append(loss.detach().to("cpu").numpy())
            with torch.no_grad():
                maes.append(
                    MAELoss(self.scaler.inverse_transform(preds), batch.y,
                            weights=batch.weight, reduction='sum').to('cpu').numpy()
                )
        train_mae = sum(maes) / len(self.train_structures)
        self.scheduler.step(train_mae)
        return train_mae, np.mean(mses)

    def evaluate_on_test(self, return_predictions=False):
        total, results = [], []
        self.model.train(False)
        with torch.no_grad():
            for batch in self.testloader:
                batch = batch.to(self.device)
                preds = self._forward(batch)
                total.append(
                    MAELoss(self.scaler.inverse_transform(preds), batch.y,
                            weights=batch.weight, reduction='sum').to('cpu').data.numpy()
                )
                results.append(self.scaler.inverse_transform(preds))
            cur_test_loss = sum(total) / len(self.test_structures)
        if not return_predictions:
            return cur_test_loss
        return cur_test_loss, torch.concat(results).to('cpu').data.reshape(-1, 1)

    def predict_structures(self, test_X, test_y, model_state_dict, test_weights=None,
                           return_predictions=False):
        print("converting data")
        test_data = [set_attr(s, y, 'y') for s, y in zip(test_X, test_y)]
        if test_weights is not None:
            test_data = [set_attr(s, w, 'weight') for s, w in zip(test_data, test_weights)]
        self.test_structures = [self.converter.convert(s) for s in tqdm(test_data)]
        loader = DataLoader(
            self.test_structures,
            batch_size=self.config["model"]["test_batch_size"],
            shuffle=False,
        )
        self.model.load_state_dict(model_state_dict)
        self.model.train(False)
        total, results = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self._forward(batch)
                preds = self.scaler.inverse_transform(preds)
                total.append(
                    MAELoss(preds, batch.y,
                            weights=batch.weight, reduction='sum').to('cpu').data.numpy()
                )
                results.append(preds.detach().to('cpu').reshape(-1, 1))
        mae = sum(total) / len(self.test_structures)
        if not return_predictions:
            return mae
        return mae, torch.concat(results).numpy().reshape(-1)

    # -------------------------------------------------------------- #
    #  Save / load
    # -------------------------------------------------------------- #

    def save(self, path):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
        }
        torch.save(state_dict, str(path) + '/checkpoint.pth')

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location)
        try:
            self.model.load_state_dict(checkpoint['model'])
        except Exception:
            print("No model parameters found")
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception:
            print("No optimizer parameters found")
        try:
            self.scaler.load_state_dict(checkpoint['scaler'])
        except Exception:
            print("No scaler parameters found")
