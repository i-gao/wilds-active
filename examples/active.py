from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.common.grouper import CombinatorialGrouper
import torch
import numpy as np
import pandas as pd
from utils import configure_split_dict
from dataset_modifications import fmow_deduplicate_locations
from copy import copy
from train import save_pseudo_if_needed
        
def run_active_learning(selection_fn, datasets, grouper, config, general_logger, full_dataset=None):
    label_manager = datasets[config.target_split]['label_manager']

    # Add labeled test / unlabeled test splits.
    labeled_split_name = f"labeled_{config.target_split}_joint" if config.concat_source_labeled else f"labeled_{config.target_split}"
    
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

    # Add new splits to datasets dict
    ## Training Splits
    ### Labeled test
    datasets[labeled_split_name] = configure_split_dict(
        data=labeled_dataset,
        split=labeled_split_name,
        split_name=labeled_split_name,
        get_train=True,
        verbose=True,
        grouper=labeled_grouper,
        batch_size=config.batch_size,
        config=labeled_config)
    ### Unlabeled test
    datasets[f'unlabeled_{config.target_split}_augmented'] = configure_split_dict(
        data=label_manager.get_unlabeled_subset(train=True),
        split=f"unlabeled_{config.target_split}_augmented",
        split_name=f"unlabeled_{config.target_split}_augmented",
        get_train=True,
        get_eval=True,
        grouper=grouper,
        batch_size=config.unlabeled_batch_size,
        verbose=True,
        config=config)
    ## Eval Splits
    ### Unlabeled test, eval transform
    datasets[f'unlabeled_{config.target_split}'] = configure_split_dict(
        data=label_manager.get_unlabeled_subset(train=False, return_pseudolabels=False),
        split=f"unlabeled_{config.target_split}",
        split_name=f"unlabeled_{config.target_split}",
        get_eval=True,
        grouper=None,
        verbose=True,
        batch_size=config.unlabeled_batch_size,
        config=config)
    
    ## Special de-duplicated eval set for fmow
    if config.dataset == 'fmow':
        disjoint_unlabeled_indices = fmow_deduplicate_locations(
            negative_indices=label_manager.labeled_indices, 
            superset_indices=label_manager.unlabeled_indices, 
            config=config)
        # dump indices to file
        pd.DataFrame(disjoint_unlabeled_indices).to_csv(f'{config.log_dir}/disjoint_indices.csv', index=False, header=False)
        # build disjoint split        
        disjoint_eval_dataset = WILDSSubset(
            full_dataset,
            disjoint_unlabeled_indices,
            label_manager.eval_transform
        )
        datasets[f'unlabeled_{config.target_split}_disjoint'] = configure_split_dict(
            data=disjoint_eval_dataset,
            split=f'unlabeled_{config.target_split}_disjoint',
            split_name=f'unlabeled_{config.target_split}_disjoint',
            get_eval=True,
            grouper=None,
            verbose=True,
            batch_size=config.unlabeled_batch_size,
            config=config)

    # Save NoisyStudent pseudolabels initially
    if config.algorithm == 'NoisyStudent':
        save_pseudo_if_needed(label_manager.unlabeled_pseudolabel_array, datasets[f'unlabeled_{config.target_split}'], None, config, None)
        if f'unlabeled_{config.target_split}_disjoint' in datasets: 
            save_pseudo_if_needed(
                label_manager.unlabeled_pseudolabel_array[[label_manager.unlabeled_indices.index(i) for i in disjoint_unlabeled_indices]], 
                datasets[f'unlabeled_{config.target_split}_disjoint'],
                None, config, None)

    # return names of train_split, unlabeled_split
    return labeled_split_name, f"unlabeled_{config.target_split}_augmented"
