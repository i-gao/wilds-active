from wilds.datasets.wilds_dataset import WILDSSubset
from wilds.common.grouper import CombinatorialGrouper
import torch
from torch.utils.data import Subset
import numpy as np
from utils import configure_split_dict, configure_loaders, get_indices, PseudolabeledSubset
from train import train
from copy import copy

class LabelManager:
    """
    Wraps a WILDS subset (e.g. ood test) with the ability to reveal / hide 
    labels and index these examples separately.
    Args:
        - subset: subset to hide/reveal labels for
        - train_transform: transform to apply to labeled examples (and unlabeled examples if subset is called with train=True)
        - eval_transform: transform to apply to unlabeled examples (if subset is called with train=False)
        - unlabeled_train_transform: transform to apply to unlabeled examples (if subset is called with train=True); overwrites 
            default train_transform. This is helpful for FixMatch.
    """
    def __init__(self, subset, train_transform, eval_transform, verbose=True, unlabeled_train_transform=None, pseudolabels=None):
        self.dataset = subset.dataset
        self._pseudolabels = pseudolabels
        
        self.labeled_train_transform = train_transform
        self.unlabeled_train_transform = unlabeled_train_transform if unlabeled_train_transform is not None else train_transform
        self.eval_transform = eval_transform

        self._idx = set(subset.indices)
        self._idx_labels_revealed = set()
        self._idx_to_pos = {idx:p for p, idx in enumerate(subset.indices)}
        self.verbose = verbose

    def get_unlabeled_subset(self, train=False):
        if train:
            subset = WILDSSubset(self.dataset, self.unlabeled_indices, self.unlabeled_train_transform)
            if self._pseudolabels is not None: 
                return PseudolabeledSubset(subset, self.unlabeled_pseudolabel_array)
            else: return subset
        else:
            return WILDSSubset(self.dataset, self.unlabeled_indices, self.eval_transform)

    def get_labeled_subset(self):
        return WILDSSubset(self.dataset, self.labeled_indices, self.labeled_train_transform)

    def reveal_labels(self, idx: [int]):
        """Remembers these examples as having labels revealed, 
        so these examples will be returned via a different data loader"""
        if not (set(idx).issubset(self._idx)):
            raise ValueError('Some indices are invalid.')
        if len(set(idx).intersection(self._idx_labels_revealed)): 
            raise ValueError('Some indices have already been selected.')
        self._idx_labels_revealed.update(idx)
        if self.verbose: print(f"Total Labels Revealed: {len(self._idx_labels_revealed)}")

    def hide_labels(self, idx: [int]):
        """Unremembers revealed example labels, undoing reveal_labels()"""
        if not (set(idx).issubset(self._idx)):
            raise ValueError('Some indices are invalid.')
        if not (set(idx).issubset(self._idx_labels_revealed)):
            raise ValueError('Some indices were never revealed.')
        self._idx_labels_revealed = self._idx_labels_revealed - set(idx)
        if self.verbose: print(f"Total Labels Revealed: {len(self._idx_labels_revealed)}")

    @property
    def num_labeled(self):
        return len(self._idx_labels_revealed)
    
    @property
    def num_unlabeled(self):
        return len(self._idx - self._idx_labels_revealed)

    @property
    def labeled_indices(self):
        return list(self._idx_labels_revealed) 

    @property
    def unlabeled_indices(self):
        return list(self._idx - self._idx_labels_revealed)
    
    @property
    def unlabeled_metadata_array(self):
        return self.get_unlabeled_subset().metadata_array
    
    @property
    def labeled_metadata_array(self):
        return self.get_labeled_subset().metadata_array

    @property
    def unlabeled_y_array(self):
        return self.get_unlabeled_subset().y_array
    
    @property
    def unlabeled_pseudolabel_array(self):
        if self._pseudolabels is not None:
            return [self._pseudolabels[self._idx_to_pos[i]] for i in self.unlabeled_indices]
        else:
            raise Exception("No pseudolabels were provided to the label maanger.")
    
    @property
    def labeled_y_array(self):
        return self.get_labeled_subset().y_array
        
def run_active_learning(selection_fn, algorithm, datasets, general_logger, grouper, config, full_dataset=None):
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

    for rnd in range(config.n_rounds):
        general_logger.write('\nActive Learning Round [%d]:\n' % rnd)
        
        # First run selection function
        selection_fn.select_and_reveal(label_manager=label_manager, K=config.n_shots)
        
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

        # Then few-shot train on the new labels
        train(
            algorithm=algorithm,
            datasets=datasets,
            train_split=labeled_split_name,
            val_split="val",
            unlabeled_split=f"unlabeled_{config.target_split}_shuffled",
            general_logger=general_logger,
            config=config,
            rnd=rnd,
            epoch_offset=0,
            best_val_metric=None)

