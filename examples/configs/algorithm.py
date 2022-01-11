algorithm_defaults = {
    'ERM': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'randaugment_n': 2,     # If we are running ERM + data augmentation
    },
    'groupDRO': {
        'train_loader': 'standard',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'group_dro_step_size': 0.01,
    },
    'deepCORAL': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'coral_penalty_weight': 1.,
    },
    'IRM': {
        'train_loader': 'group',
        'uniform_over_groups': True,
        'distinct_groups': True,
        'eval_loader': 'standard',
        'irm_lambda': 100.,
        'irm_penalty_anneal_iters': 500,
    },
    'MAML': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'maml_first_order': True,
        'metalearning_kwargs': {
            'n_adapt_steps': 5,
            'n_eval_examples': 16
        }
    },
    'ANIL': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'metalearning_kwargs': {
            'n_adapt_steps': 5,
            'n_eval_examples': 16
        }
    },
    'FixMatch': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'self_training_labeled_weight': 1,
        'self_training_unlabeled_weight': 1,
        'self_training_threshold': 0.7,
        # 'scheduler': 'FixMatchLR',
        # 'scheduler_kwargs': {},
        'randaugment_n': 2,
        'additional_labeled_transform': 'weak',     # Apply weak augmentation to labeled examples
        'additional_unlabeled_transform': ['weak', 'randaugment'],  # pseudolabels generated on weak, prediction on strong
    },
    'PseudoLabel': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'self_training_labeled_weight': 1,
        'self_training_unlabeled_weight': 1,
        'self_training_threshold': 0.7,
        'pseudolabel_T2': 0,                       # No scheduling
        # 'pseudolabel_T2': 0.4,
        'randaugment_n': 2, 
        'additional_labeled_transform': None,     # Normally, no augmentations
        'additional_unlabeled_transform': None,     # Normally, no augmentations
    },
    'NoisyStudent': {
        'train_loader': 'standard',
        'uniform_over_groups': False,
        'eval_loader': 'standard',
        'dropout_rate': 0.5,
        'randaugment_n': 2,
        'additional_labeled_transform': 'randaugment',     # Apply strong augmentation to labeled examples
        'additional_unlabeled_transform': 'randaugment',     # Apply strong augmentation to unlabeled examples
    }
}
