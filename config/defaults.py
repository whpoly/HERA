"""Default configuration generators for each dataset."""


def _base_optim():
    return {
        'factor': 0.9,
        'patience': 10,
        'threshold': 0.0005,
        'min_lr': 1e-5,
        'lr_initial': 1e-3,
        'scheduler': 'ReduceLROnPlateau',
    }


def get_configs_2dmd(task_prefix):
    """Configs for vacancy and 2dmd_high datasets."""
    config_sparse = {
        'task': f'{task_prefix}_sparse',
        'model': {
            'train_batch_size': 50,
            'test_batch_size': 100,
            'add_z_bond_coord': False,
            'atom_features': 'werespecies',
            'state_input_shape': 2,
            'cutoff': 12,
            'edge_embed_size': 40,
            'vertex_aggregation': 'max',
            'global_aggregation': 'max',
            'embedding_size': 64,
            'nblocks': 3,
        },
        'optim': _base_optim(),
    }
    config_full = {
        'task': f'{task_prefix}_full',
        'model': {
            'train_batch_size': 50,
            'test_batch_size': 100,
            'add_z_bond_coord': False,
            'atom_features': 'Z',
            'state_input_shape': 2,
            'cutoff': 6,
            'edge_embed_size': 40,
            'vertex_aggregation': 'sum',
            'global_aggregation': 'mean',
            'embedding_size': 64,
            'nblocks': 3,
        },
        'optim': _base_optim(),
    }
    config_hetero = {
        'task': f'{task_prefix}_hetero',
        'model': {
            'train_batch_size': 50,
            'test_batch_size': 100,
            'add_z_bond_coord': False,
            'atom_features': 'Z',
            'state_input_shape': 2,
            'cutoff': 6,
            'edge_embed_size': 40,
            'vertex_aggregation': 'sum',
            'global_aggregation': 'mean',
            'embedding_size': 64,
            'nblocks': 3,
        },
        'optim': _base_optim(),
    }
    config_attention = {
        'task': f'{task_prefix}_attention',
        'model': {
            'train_batch_size': 50,
            'test_batch_size': 100,
            'add_z_bond_coord': False,
            'atom_features': 'Z',
            'state_input_shape': 2,
            'cutoff': 6,
            'edge_embed_size': 40,
            'vertex_aggregation': 'sum',
            'global_aggregation': 'mean',
            'embedding_size': 64,
            'nblocks': 3,
            'n_heads': 4,
        },
        'optim': _base_optim(),
    }
    return config_sparse, config_full, config_hetero, config_attention


def get_configs_default(task_prefix):
    """Default configs for native, och, imp2d, semi datasets."""
    config_sparse = {
        'task': f'{task_prefix}_sparse',
        'model': {
            'train_batch_size': 50,
            'test_batch_size': 100,
            'add_z_bond_coord': False,
            'atom_features': 'Z',
            'state_input_shape': 2,
            'cutoff': 6,
            'edge_embed_size': 40,
            'vertex_aggregation': 'sum',
            'global_aggregation': 'mean',
            'embedding_size': 64,
            'nblocks': 3,
        },
        'optim': _base_optim(),
    }
    config_full = {
        'task': f'{task_prefix}_full',
        'model': {
            'train_batch_size': 50,
            'test_batch_size': 100,
            'add_z_bond_coord': False,
            'atom_features': 'Z',
            'state_input_shape': 2,
            'cutoff': 6,
            'edge_embed_size': 40,
            'vertex_aggregation': 'sum',
            'global_aggregation': 'mean',
            'embedding_size': 64,
            'nblocks': 3,
        },
        'optim': _base_optim(),
    }
    config_hetero = {
        'task': f'{task_prefix}_hetero',
        'model': {
            'train_batch_size': 50,
            'test_batch_size': 100,
            'add_z_bond_coord': False,
            'atom_features': 'Z',
            'state_input_shape': 2,
            'cutoff': 6,
            'edge_embed_size': 40,
            'vertex_aggregation': 'sum',
            'global_aggregation': 'mean',
            'embedding_size': 64,
            'nblocks': 3,
        },
        'optim': _base_optim(),
    }
    config_attention = {
        'task': f'{task_prefix}_attention',
        'model': {
            'train_batch_size': 50,
            'test_batch_size': 100,
            'add_z_bond_coord': False,
            'atom_features': 'Z',
            'state_input_shape': 2,
            'cutoff': 6,
            'edge_embed_size': 40,
            'vertex_aggregation': 'sum',
            'global_aggregation': 'mean',
            'embedding_size': 64,
            'nblocks': 3,
            'n_heads': 4,
        },
        'optim': _base_optim(),
    }
    return config_sparse, config_full, config_hetero, config_attention


# Maps dataset name -> config generator
_CONFIG_REGISTRY = {
    'vacancy': get_configs_2dmd,
    '2dmd_high': get_configs_2dmd,
    'native': get_configs_default,
    'och': get_configs_default,
    'imp2d': get_configs_default,
    'semi': get_configs_default,
}

VALID_DATASETS = list(_CONFIG_REGISTRY.keys())
VALID_MODELS = ['megnet', 'cgcnn']
VALID_MODES = ['sparse', 'full', 'hetero', 'attention']


def get_config(model: str, dataset: str, mode: str):
    """Get the config dict for a specific model/dataset/mode combination.

    Args:
        model: 'megnet' or 'cgcnn'
        dataset: one of VALID_DATASETS
        mode: one of 'sparse', 'full', 'hetero', 'attention'

    Returns:
        config dict ready for MEGNetTrainer
    """
    if dataset not in _CONFIG_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {VALID_DATASETS}")
    if model not in VALID_MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {VALID_MODELS}")
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {VALID_MODES}")

    config_sparse, config_full, config_hetero, config_attention = _CONFIG_REGISTRY[dataset](model)
    return {'sparse': config_sparse, 'full': config_full,
            'hetero': config_hetero, 'attention': config_attention}[mode]
