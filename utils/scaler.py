"""Scaler and helper tensor types."""

import numpy as np
import torch
from copy import copy


class MyTensor(torch.Tensor):
    """Tensor subclass that returns 0 for max() on empty tensors (needed for edgeless graphs)."""

    def max(self, *args, **kwargs):
        if torch.numel(self) == 0:
            return 0
        else:
            return torch.max(self, *args, **kwargs)


class Scaler:
    """Simple z-score scaler for target values."""

    def __init__(self):
        self.mean = 0
        self.std = 1.0

    def fit(self, dataset, feature_name='y'):
        data = np.array([getattr(dataset[i], feature_name) for i in range(len(dataset))])
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        data_copy = copy(data)
        return (data_copy - self.mean) / (self.std if abs(self.std) > 1e-7 else 1.)

    def inverse_transform(self, data):
        data_copy = data.detach().clone()
        std = self.std if abs(self.std) > 1e-7 else 1.0
        return data_copy * std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
