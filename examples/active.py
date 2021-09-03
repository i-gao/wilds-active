from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.common.grouper import CombinatorialGrouper
import torch
import numpy as np
from utils import configure_split_dict, configure_loaders
from copy import copy
        
def run_active_learning(selection_fn, datasets, grouper, config, general_logger, full_dataset=None):
    label_manager = datasets[config.target_split]['label_manager']

    # Add labeled test / unlabeled test splits.
    labeled_split_name = f"labeled_{config.target_split}_joint" if config.concat_source_labeled else f"labeled_{config.target_split}"
    datasets[labeled_split_name] = configure_split_dict( # labeled for training
        data=None,
        split=labeled_split_name,
        split_name=labeled_split_name,
        train=True,
        verbose=True,
        grouper=grouper,
        config=config)
    datasets[f'unlabeled_{config.target_split}_shuffled'] = configure_split_dict( # unlabeled for training
        data = None,
        split=f"unlabeled_{config.target_split}_shuffled",
        split_name=f"unlabeled_{config.target_split}_shuffled",
        train=False,
        grouper=None,
        verbose=True,
        config=config)
    datasets[f'unlabeled_{config.target_split}'] = configure_split_dict( # unlabeled for eval
        data = None,
        split=f"unlabeled_{config.target_split}",
        split_name=f"unlabeled_{config.target_split}",
        train=False,
        grouper=None,
        verbose=True,
        config=config)
    
    # First run selection function
    selection_fn.select_and_reveal(label_manager=label_manager, K=config.n_shots)
    general_logger.write(f"Total Labels Revealed: {label_manager.num_labeled}\n")
    
    # Concatenate labeled source examples to labeled target examples
    if config.concat_source_labeled:
        assert full_dataset is not None
        labeled_dataset = WILDSSubset(
            full_dataset,
            np.concatenate((label_manager.labeled_indices, datasets['train']['dataset'].indices)).astype(int), # target points at front
            label_manager.labeled_train_transform
        ) 
    else:
        labeled_dataset = label_manager.get_labeled_subset()

    if config.upsample_target_labeled:
        # upsample target labels (compared to src labels) using a weighted sampler
        # do this by grouping by split and then using --uniform_over_groups=True
        labeled_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['split']
        )
        labeled_config=copy(config)
        labeled_config.uniform_over_groups = True
    else:
        labeled_config = config
        labeled_grouper = grouper

    configure_loaders( # labeled for training
        split_dict=datasets[labeled_split_name],
        data=labeled_dataset,
        train=True,
        grouper=labeled_grouper,
        batch_size=config.batch_size,
        config=labeled_config)
    configure_loaders( # unlabeled for training
        split_dict=datasets[f'unlabeled_{config.target_split}_shuffled'],
        data=label_manager.get_unlabeled_subset(train=True),
        train=True,
        grouper=grouper,
        batch_size=config.unlabeled_batch_size, # special batch size
        config=config)
    configure_loaders( # for eval
        split_dict=datasets[f'unlabeled_{config.target_split}'],
        data=label_manager.get_unlabeled_subset(),
        train=False,
        grouper=None,
        batch_size=config.batch_size,
        config=config)
    
    # return names of train_split, unlabeled_split
    return labeled_split_name, f"unlabeled_{config.target_split}_shuffled"

