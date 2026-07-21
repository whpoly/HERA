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
    config_was_x = _was_config(config_full, f'{task_prefix}_was_x')
    config_hetero_was = _was_config(config_hetero, _hetero_was_task(task_prefix))
    config_attention_was = _was_config(config_attention, f'{task_prefix}_attention_was')
    return (
        config_full,
        config_hetero,
        config_attention,
        config_was_x,
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
    config_was_x = _was_config(config_full, f'{task_prefix}_was_x')
    config_hetero_was = _was_config(config_hetero, _hetero_was_task(task_prefix))
    config_attention_was = _was_config(config_attention, f'{task_prefix}_attention_was')
    return (
        config_full,
        config_hetero,
        config_attention,
        config_was_x,
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
VALID_MODELS = ['alignn', 'megnet', 'cgcnn', 'definet']
ALIGNN_TRAIN_BATCH_SIZE = 64
ALIGNN_TEST_BATCH_SIZE = 1
ALIGNN_BLOCKS = 3
ALIGNN_GCN_BLOCKS = 3
ALIGNN_MAX_NEIGHBORS = 12
DEFINET_MODES = ('attention', 'attention_was')
ALIGNN_MODES = (
    'full',
    'full_x',
    'hetero',
    'hetero_global',
    'hetero_fixed_pool',
    'attention',
    'was_x',
    'hetero_was',
    'attention_was',
    'definet',
    'definet_was',
)
FIXED_POOL_MODES = (
    'hetero_fixed_pool',
)
ALIGNN_GLOBAL_HETERO_MODES = (
    'hetero_global',
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
    'hetero_global',
    'hetero_fixed_pool',
    'attention',
    'was_x',
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


def _finalize_config(config, model):
    """Apply model-specific defaults that differ from the shared base configs."""
    config = copy.deepcopy(config)
    if model == 'alignn':
        config['model']['nblocks'] = ALIGNN_BLOCKS
        config['model']['gcn_blocks'] = ALIGNN_GCN_BLOCKS
        config['model']['max_neighbors'] = ALIGNN_MAX_NEIGHBORS
        # Keep ALIGNN defaults close to the original training setup while using
        # a single validation/test graph to limit line-graph memory.
        config['model']['train_batch_size'] = ALIGNN_TRAIN_BATCH_SIZE
        config['model']['test_batch_size'] = ALIGNN_TEST_BATCH_SIZE
    return config


def get_config(model: str, dataset: str, mode: str):
    """Get the config dict for a specific model/dataset/mode combination.

    Args:
        model: 'megnet', 'cgcnn', 'definet', or 'alignn'
        dataset: one of VALID_DATASETS
        mode: one of 'full', 'full_x', 'hetero', 'hetero_global',
            'hetero_fixed_pool',
            'attention', 'was_x', 'hetero_was', 'attention_was',
            'definet', 'definet_was'

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
    if mode in ALIGNN_GLOBAL_HETERO_MODES and model != 'alignn':
        raise ValueError("The hetero_global mode is only supported for alignn")
    if mode in FIXED_POOL_MODES and model not in ('cgcnn', 'megnet', 'alignn'):
        raise ValueError("The hetero_fixed_pool mode is only supported for cgcnn, megnet, and alignn")
    if model not in WAS_MODELS and mode in (
            'was_x',
            'hetero_was',
    ):
        raise ValueError("The was_x and hetero_was modes are only supported for cgcnn, megnet, and alignn")
    if model not in ATTENTION_ABLATION_MODELS and mode in (
            'attention_was',
    ):
        raise ValueError("The attention ablation modes are only supported for cgcnn, megnet, definet, and alignn")

    (
        config_full,
        config_hetero,
        config_attention,
        config_was_x,
        config_hetero_was,
        config_attention_was,
    ) = _CONFIG_REGISTRY[dataset](model)
    if mode in CGCNN_DEFINET_MODES:
        return _finalize_config(_definet_attention_config(config_attention, mode, model), model)
    if mode == 'full_x':
        config = copy.deepcopy(config_full)
        config['task'] = f'{model}_full_x'
        return _finalize_config(config, model)
    if mode == 'hetero_fixed_pool':
        config = copy.deepcopy(config_hetero)
        config['task'] = f'{model}_hetero_fixed_pool'
        config['model']['fixed_pooling'] = True
        return _finalize_config(config, model)
    if mode == 'hetero_global':
        config = copy.deepcopy(config_hetero)
        config['task'] = 'alignn_hetero_global'
        config['model']['use_global_node'] = True
        return _finalize_config(config, model)
    if mode == 'was_x':
        config = copy.deepcopy(config_was_x)
        config['task'] = f'{model}_was_x'
        return _finalize_config(config, model)

    config = {'full': config_full,
              'hetero': config_hetero,
              'attention': config_attention,
              'was_x': config_was_x,
              'hetero_was': config_hetero_was,
              'attention_was': config_attention_was,
              }[mode]
    if model == 'definet':
        config['model']['nblocks'] = 4
        config['model']['n_marker_types'] = 2
        config['model'].pop('n_heads', None)
    return _finalize_config(config, model)
