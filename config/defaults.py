"""Default configuration generators for each dataset."""

import copy


def _base_optim():
    return {
        'factor': 0.9,
        'patience': 10,
        'threshold': 0.0005,
        'min_lr': 1e-5,
        'lr_initial': 1e-3,
        'scheduler': 'ReduceLROnPlateau',
    }


def _was_config(base_config, task):
    config = copy.deepcopy(base_config)
    config['task'] = task
    config['model']['atom_features'] = 'was_species'
    return config


def _hetero_was_task(task_prefix):
    if task_prefix == 'cgcnn':
        return 'hetero_cgcnn_was'
    return f'{task_prefix}_hetero_was'


def get_configs_2dmd(task_prefix):
    """Configs for vacancy and 2dmd_high datasets."""
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
    config_was = _was_config(config_full, f'{task_prefix}_was')
    config_hetero_was = _was_config(config_hetero, _hetero_was_task(task_prefix))
    config_attention_was = _was_config(config_attention, f'{task_prefix}_attention_was')
    return (
        config_full,
        config_hetero,
        config_attention,
        config_was,
        config_hetero_was,
        config_attention_was,
    )


def get_configs_default(task_prefix):
    """Default configs for native, och, imp2d, semi datasets."""
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
    config_was = _was_config(config_full, f'{task_prefix}_was')
    config_hetero_was = _was_config(config_hetero, _hetero_was_task(task_prefix))
    config_attention_was = _was_config(config_attention, f'{task_prefix}_attention_was')
    return (
        config_full,
        config_hetero,
        config_attention,
        config_was,
        config_hetero_was,
        config_attention_was,
    )


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
VALID_MODELS = ['megnet', 'cgcnn', 'definet', 'alignn']
DEFINET_MODES = ('attention', 'attention_was')
ALIGNN_MODES = (
    'full',
    'full_x',
    'hetero',
    'attention',
    'was',
    'hetero_was',
    'attention_was',
    'definet',
    'definet_was',
)
CGCNN_DEFINET_MODES = (
    'definet',
    'definet_was',
)
CGCNN_DEFINET_TASKS = {
    'definet': 'definet_attention',
    'definet_was': 'definet_attention_was',
}
ALIGNN_DEFINET_TASKS = {
    'definet': 'alignn_definet',
    'definet_was': 'alignn_definet_was',
}
WAS_MODELS = ('cgcnn', 'megnet', 'alignn')
ATTENTION_ABLATION_MODELS = ('cgcnn', 'megnet', 'definet', 'alignn')
VALID_MODES = [
    'full',
    'full_x',
    'hetero',
    'attention',
    'was',
    'hetero_was',
    'attention_was',
    'definet',
    'definet_was',
]


def _definet_attention_config(base_config, mode, model='cgcnn'):
    config = copy.deepcopy(base_config)
    if model == 'alignn':
        config['task'] = ALIGNN_DEFINET_TASKS[mode]
    else:
        config['task'] = CGCNN_DEFINET_TASKS[mode]
    if mode == 'definet_was':
        config['model']['atom_features'] = 'was_species'
    config['model']['nblocks'] = 4
    config['model']['n_marker_types'] = 2
    config['model'].pop('n_heads', None)
    return config


def get_config(model: str, dataset: str, mode: str):
    """Get the config dict for a specific model/dataset/mode combination.

    Args:
        model: 'megnet', 'cgcnn', 'definet', or 'alignn'
        dataset: one of VALID_DATASETS
        mode: one of 'full', 'full_x', 'hetero', 'attention', 'was',
            'hetero_was', 'attention_was', 'definet', 'definet_was'

    Returns:
        config dict ready for MEGNetTrainer
    """
    if dataset not in _CONFIG_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {VALID_DATASETS}")
    if model not in VALID_MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {VALID_MODELS}")
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {VALID_MODES}")
    if mode in CGCNN_DEFINET_MODES and model not in ('cgcnn', 'alignn'):
        raise ValueError("The definet modes are run under --model cgcnn or --model alignn")
    if model == 'definet' and mode not in DEFINET_MODES:
        raise ValueError(f"The definet model only supports {DEFINET_MODES}")
    if model == 'alignn' and mode not in ALIGNN_MODES:
        raise ValueError(f"The alignn model only supports {ALIGNN_MODES}")
    if model not in WAS_MODELS and mode in (
            'was',
            'hetero_was',
    ):
        raise ValueError("The was and hetero_was modes are only supported for cgcnn, megnet, and alignn")
    if model not in ATTENTION_ABLATION_MODELS and mode in (
            'attention_was',
    ):
        raise ValueError("The attention ablation modes are only supported for cgcnn, megnet, definet, and alignn")

    (
        config_full,
        config_hetero,
        config_attention,
        config_was,
        config_hetero_was,
        config_attention_was,
    ) = _CONFIG_REGISTRY[dataset](model)
    if mode in CGCNN_DEFINET_MODES:
        return _definet_attention_config(config_attention, mode, model)
    if mode == 'full_x':
        config = copy.deepcopy(config_full)
        config['task'] = f'{model}_full_x'
        return config

    config = {'full': config_full,
              'hetero': config_hetero,
              'attention': config_attention,
              'was': config_was,
              'hetero_was': config_hetero_was,
              'attention_was': config_attention_was,
              }[mode]
    if model == 'definet':
        config['model']['nblocks'] = 4
        config['model']['n_marker_types'] = 2
        config['model'].pop('n_heads', None)
    return config
