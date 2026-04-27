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
from ..utils.scaler import Scaler
from .losses import MAELoss


def set_attr(structure, attr, name):
    setattr(structure, name, attr)
    return structure


class MEGNetTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device

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
            add_z_bond_coord=self.config["model"]["add_z_bond_coord"],
            add_eos_features=(use_eos := self.config["model"].get("add_eos_features", False)),
        )
        self.scaler = Scaler()

        task = self.config['task']
        # Build model based on task string:  {model}_{mode}
        if task == 'megnet_full' or task == 'megnet_sparse':
            self.model = MEGNet(
                edge_input_shape=bond_converter.get_shape(eos=use_eos),
                node_input_shape=atom_converter.get_shape(),
                embedding_size=self.config['model']['embedding_size'],
                n_blocks=self.config['model']['nblocks'],
                state_input_shape=self.config["model"]["state_input_shape"],
                vertex_aggregation=self.config["model"]["vertex_aggregation"],
                global_aggregation=self.config["model"]["global_aggregation"],
            ).to(self.device)
        elif task == 'megnet_hetero' or task == 'megnet_local':
            self.model = HeteroMEGNet(
                edge_input_shape=bond_converter.get_shape(eos=use_eos),
                node_input_shape=None,
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
            ).to(self.device)
        elif task == 'cgcnn_hetero' or task == 'cgcnn_local':
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
            ).to(self.device)
        elif task == 'megnet_attention':
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
        elif task == 'cgcnn_attention':
            self.model = AttentionCGCNN(
                orig_atom_fea_len=atom_converter.get_shape(),
                nbr_fea_len=bond_converter.get_shape(eos=use_eos),
                n_conv=self.config['model']['nblocks'],
                n_heads=self.config['model'].get('n_heads', 4),
                n_h=3,
            ).to(self.device)
        elif task == 'definet_attention':
            self.model = DefiNet(
                orig_atom_fea_len=atom_converter.get_shape(),
                nbr_fea_len=bond_converter.get_shape(eos=use_eos),
                atom_fea_len=self.config['model']['embedding_size'],
                n_conv=self.config['model']['nblocks'],
                n_marker_types=self.config['model'].get('n_marker_types', 2),
                n_h=3,
            ).to(self.device)
        else:
            # default: cgcnn_full / cgcnn_sparse
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
                verbose=True,
            )
        else:
            raise ValueError("Unknown scheduler")

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
            self.sampler = WeightedRandomSampler(torch.tensor(train_weights).float(), len(train_weights))

        self.trainloader = DataLoader(
            self.train_structures,
            batch_size=self.config["model"]["train_batch_size"],
            shuffle=False if train_weights is not None else True,
            num_workers=0,
            sampler=self.sampler,
        )
        self.testloader = DataLoader(
            self.test_structures,
            batch_size=self.config["model"]["test_batch_size"],
            shuffle=False,
            num_workers=0,
        )
        self.target_name = target_name

    # -------------------------------------------------------------- #
    #  Forward helper (dispatches by task)
    # -------------------------------------------------------------- #

    def _forward(self, batch):
        task = self.config['task']
        if task in ('megnet_hetero', 'megnet_local'):
            return self.model(
                batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict,
                batch.state, batch.batch_dict, batch.bond_batch_dict,
            ).squeeze()
        elif task in ('cgcnn_hetero', 'cgcnn_local'):
            return self.model(
                batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict,
                batch.batch_dict,
            ).squeeze()
        elif task in ('cgcnn_sparse', 'cgcnn_full'):
            return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
        elif task == 'cgcnn_attention':
            return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, node_type=batch.node_type).squeeze()
        elif task == 'definet_attention':
            marker = getattr(batch, 'defect_marker', getattr(batch, 'node_type', None))
            return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, defect_marker=marker).squeeze()
        elif task == 'megnet_attention':
            return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch, node_type=batch.node_type).squeeze()
        else:
            return self.model(batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch).squeeze()

    # -------------------------------------------------------------- #
    #  Training / evaluation
    # -------------------------------------------------------------- #

    def train_one_epoch(self):
        mses, maes = [], []
        self.model.train(True)
        for batch in self.trainloader:
            batch = batch.to(self.device)
            preds = self._forward(batch)
            loss = F.mse_loss(self.scaler.transform(batch.y), preds, reduction='mean')
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            mses.append(loss.to("cpu").data.numpy())
            maes.append(
                MAELoss(self.scaler.inverse_transform(preds), batch.y,
                        weights=batch.weight, reduction='sum').to('cpu').data.numpy()
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

    def predict_structures(self, test_X, test_y, model_state_dict, test_weights=None):
        print("converting data")
        test_data = [set_attr(s, y, 'y') for s, y in zip(test_X, test_y)]
        if test_weights is not None:
            test_data = [set_attr(s, w, 'weight') for s, w in zip(test_data, test_weights)]
        self.test_structures = [self.converter.convert(s) for s in tqdm(test_data)]
        loader = DataLoader(self.test_structures, batch_size=50, shuffle=False)
        self.model.load_state_dict(model_state_dict)
        self.model.train(False)
        total = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self._forward(batch)
                total.append(
                    MAELoss(self.scaler.inverse_transform(preds), batch.y,
                            weights=batch.weight, reduction='sum').to('cpu').data.numpy()
                )
        return sum(total) / len(self.test_structures)

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
