"""Default configuration generators for each dataset."""

import copy


EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_MIN_DELTA_PERCENT = 0.5
HYPERGRAPH_SCHEMA = 'per_defect_neighborhood_v2'


def _base_optim():
    return {
        'optimizer': 'AdamW',
        'weight_decay': 1e-4,
        'factor': 0.9,
        'patience': 10,
        'threshold': 0.0005,
        'min_lr': 1e-5,
        'lr_initial': 1e-3,
        'scheduler': 'ReduceLROnPlateau',
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'early_stopping_min_delta_percent': EARLY_STOPPING_MIN_DELTA_PERCENT,
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
    """Configs for vacancy, 2dmd_low, and 2dmd_high datasets."""
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
    config_was_x = _was_config(config_full, f'{task_prefix}_was_x')
    config_hetero_was = _was_config(config_hetero, _hetero_was_task(task_prefix))
    config_attention_was = _was_config(config_attention, f'{task_prefix}_attention_was')
    return (
        config_sparse,
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
        None,
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
    '2dmd_low': get_configs_2dmd,
    '2dmd_high': get_configs_2dmd,
    'native': get_configs_default,
    'och': get_configs_default,
    'imp2d': get_configs_default,
    'semi': get_configs_default,
}

VALID_DATASETS = list(_CONFIG_REGISTRY.keys())
VALID_MODELS = ['alignn', 'megnet', 'cgcnn', 'definet', 'hypergraph']
VACANCY_TRAIN_BATCH_SIZE = 8
MEMORY_LIMITED_TRAIN_BATCH_SIZE = 16
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_TEST_BATCH_SIZE = 1
CGCNN_MEGNET_MAX_NEIGHBORS = 12
MEGNET_EMBEDDING_SIZE = 32
MEGNET_HETERO_EMBEDDING_SIZE = 32
ALIGNN_BLOCKS = 3
ALIGNN_GCN_BLOCKS = 3
ALIGNN_MAX_NEIGHBORS = 12
ALIGNN_HETERO_NODE_NORM = 'layernorm'
DEFINET_MODES = ('attention', 'attention_was')
ALIGNN_MODES = (
    'full',
    'full_x',
    'hetero',
    'hetero_fixed_pool',
    'attention',
    'was_x',
    'hetero_was',
    'attention_was',
    'definet',
    'definet_was',
    'hypergraph',
    'hypergraph_was',
)
FIXED_POOL_MODES = (
    'hetero_fixed_pool',
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
HYPERGRAPH_MODELS = ('cgcnn', 'megnet', 'alignn', 'hypergraph')
VALID_MODES = [
    'sparse',
    'full',
    'full_x',
    'hetero',
    'hetero_fixed_pool',
    'attention',
    'was_x',
    'hetero_was',
    'attention_was',
    'definet',
    'definet_was',
    'hypergraph',
    'hypergraph_was',
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


def _finalize_config(config, model, dataset):
    """Apply model- and dataset-specific defaults."""
    config = copy.deepcopy(config)
    config['model']['train_batch_size'] = DEFAULT_TRAIN_BATCH_SIZE
    config['model']['test_batch_size'] = DEFAULT_TEST_BATCH_SIZE
    if model in ('cgcnn', 'megnet'):
        config['model']['max_neighbors'] = CGCNN_MEGNET_MAX_NEIGHBORS
    if model == 'megnet':
        is_hetero = '_hetero' in config['task']
        config['model']['embedding_size'] = (
            MEGNET_HETERO_EMBEDDING_SIZE if is_hetero
            else MEGNET_EMBEDDING_SIZE
        )
    if model == 'alignn':
        config['model']['nblocks'] = ALIGNN_BLOCKS
        config['model']['gcn_blocks'] = ALIGNN_GCN_BLOCKS
        config['model']['max_neighbors'] = ALIGNN_MAX_NEIGHBORS
        config['model']['hetero_node_norm'] = ALIGNN_HETERO_NODE_NORM
    if dataset in ('vacancy', '2dmd_low'):
        config['model']['train_batch_size'] = VACANCY_TRAIN_BATCH_SIZE
    elif dataset in ('2dmd_high', 'native'):
        config['model']['train_batch_size'] = MEMORY_LIMITED_TRAIN_BATCH_SIZE
    return config


def _finalize_sparse_config(config, dataset):
    """Apply the current training protocol without changing sparse architecture."""
    config = copy.deepcopy(config)
    config['model']['test_batch_size'] = DEFAULT_TEST_BATCH_SIZE
    if dataset in ('vacancy', '2dmd_low'):
        config['model']['train_batch_size'] = VACANCY_TRAIN_BATCH_SIZE
    else:
        config['model']['train_batch_size'] = MEMORY_LIMITED_TRAIN_BATCH_SIZE
    return config


def get_config(model: str, dataset: str, mode: str):
    """Get the config dict for a specific model/dataset/mode combination.

    Args:
        model: 'megnet', 'cgcnn', 'definet', 'alignn', or 'hypergraph'
        dataset: one of VALID_DATASETS
        mode: one of 'sparse', 'full', 'full_x', 'hetero', 'hetero_fixed_pool',
            'attention', 'was_x', 'hetero_was', 'attention_was',
            'definet', 'definet_was', 'hypergraph', 'hypergraph_was'

    Returns:
        config dict ready for MEGNetTrainer
    """
    if dataset not in _CONFIG_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {VALID_DATASETS}")
    if model not in VALID_MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {VALID_MODELS}")
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {VALID_MODES}")
    if model == 'hypergraph' and mode != 'hypergraph':
        raise ValueError("The hypergraph model only supports mode 'hypergraph'")
    if mode == 'hypergraph' and model not in HYPERGRAPH_MODELS:
        raise ValueError(
            "The hypergraph mode is only supported for cgcnn, megnet, "
            "alignn, and hypergraph"
        )
    if mode == 'hypergraph_was' and model not in WAS_MODELS:
        raise ValueError(
            "The hypergraph_was mode is only supported for cgcnn, megnet, "
            "and alignn"
        )
    if mode == 'sparse' and (
            model != 'megnet'
            or dataset not in ('vacancy', '2dmd_low', '2dmd_high')
    ):
        raise ValueError(
            "The sparse mode is the MEGNET_SPARSE reproduction and is only "
            "supported for --model megnet on vacancy, 2dmd_low, or 2dmd_high"
        )
    if mode in CGCNN_DEFINET_MODES and model not in ('cgcnn', 'alignn'):
        raise ValueError("The definet modes are run under --model cgcnn or --model alignn")
    if model == 'definet' and mode not in DEFINET_MODES:
        raise ValueError(f"The definet model only supports {DEFINET_MODES}")
    if model == 'alignn' and mode not in ALIGNN_MODES:
        raise ValueError(f"The alignn model only supports {ALIGNN_MODES}")
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
        config_sparse,
        config_full,
        config_hetero,
        config_attention,
        config_was_x,
        config_hetero_was,
        config_attention_was,
    ) = _CONFIG_REGISTRY[dataset](model)
    if mode == 'sparse':
        return _finalize_sparse_config(config_sparse, dataset)
    if mode in ('hypergraph', 'hypergraph_was'):
        config = copy.deepcopy(
            config_hetero_was if mode == 'hypergraph_was' else config_hetero
        )
        config['task'] = f'{model}_{mode}'
        config['model']['hypergraph_radius'] = 3.0
        config['model']['hypergraph_schema'] = HYPERGRAPH_SCHEMA
        config['model']['n_heads'] = 4
        config['model']['dropout'] = 0.0
        return _finalize_config(config, model, dataset)
    if mode in CGCNN_DEFINET_MODES:
        return _finalize_config(
            _definet_attention_config(config_attention, mode, model),
            model,
            dataset,
        )
    if mode == 'full_x':
        config = copy.deepcopy(config_full)
        config['task'] = f'{model}_full_x'
        return _finalize_config(config, model, dataset)
    if mode == 'hetero_fixed_pool':
        config = copy.deepcopy(config_hetero)
        config['task'] = f'{model}_hetero_fixed_pool'
        config['model']['fixed_pooling'] = True
        return _finalize_config(config, model, dataset)
    if mode == 'was_x':
        config = copy.deepcopy(config_was_x)
        config['task'] = f'{model}_was_x'
        return _finalize_config(config, model, dataset)

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
    return _finalize_config(config, model, dataset)
