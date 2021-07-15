from wilds.datasets.wilds_dataset import WILDSSubset
from torch.utils.data import Subset
import numpy as np
from utils import configure_split_dict, configure_loaders, get_indices
from train import train

class LabelManager:
    """
    Wraps a WILDS subset (e.g. ood test) with the ability to reveal / hide 
    labels and index these examples separately.
    Args:
        - subset: subset to hide/reveal labels for
    """
    def __init__(self, subset: WILDSSubset, verbose=True):
        self.dataset = subset.dataset
        self.transform = subset.transform
        self._idx = set(subset.indices)
        self._idx_labels_revealed = set()
        self.verbose = verbose

    def get_unlabeled_subset(self):
        return WILDSSubset(self.dataset, self.unlabeled_indices, self.transform)

    def get_labeled_subset(self):
        return WILDSSubset(self.dataset, self.labeled_indices, self.transform)

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
    def labeled_y_array(self):
        return self.get_labeled_subset().y_array
        
def run_active_learning(selection_fn, few_shot_algorithm, datasets, general_logger, grouper, config, full_dataset=None):
    label_manager = datasets['test']['label_manager']
    joint_training = config.few_shot_kwargs.get('train_joint_source_target', False)

    # Add labeled test / unlabeled test splits.
    labeled_split_name = "labeled_test_joint" if joint_training else "labeled_test"
    datasets[labeled_split_name] = configure_split_dict(
        data=None,
        split=labeled_split_name,
        split_name=labeled_split_name,
        train=True,
        verbose=True,
        grouper=grouper,
        config=config)
    datasets['unlabeled_test'] = configure_split_dict(
        data = None,
        split="unlabeled_test",
        split_name="unlabeled_test",
        train=False,
        grouper=None,
        verbose=True,
        config=config)

    for rnd in range(config.n_rounds):
        general_logger.write('\nActive Learning Round [%d]:\n' % rnd)
        
        # First run selection function
        ## Train selection function
        # selection_fn.update()
        ## Get a few labels
        selection_fn.select_and_reveal(label_manager=label_manager, K=config.n_shots)
        
        ## Refresh dataloaders
        if joint_training:
            # Combine two WildsSubsets
            assert full_dataset is not None
            labeled_dataset = WILDSSubset(
                full_dataset,
                np.concatenate((label_manager.labeled_indices, datasets['train']['dataset'].indices)), # test points at front
                label_manager.transform
            )
            if config.few_shot_kwargs.get('single_test_group', False):
                # rig all test metadata to have the same group as the first test point
                test_metadata = labeled_dataset.metadata_array[0]
                labeled_dataset.dataset.metadata_array[label_manager.labeled_indices] = test_metadata # VERY hacky, reaches into subset's dataset's metadata array
            if config.few_shot_kwargs.get('single_train_group', False):
                # rig all train metadata to have the same group as the first train point
                train_metadata = labeled_dataset.metadata_array[-1]
                labeled_dataset.dataset.metadata_array[datasets['train']['dataset'].indices] = train_metadata # VERY hacky, reaches into subset's dataset's metadata array                
        else:
            labeled_dataset = label_manager.get_labeled_subset()

        configure_loaders(
            split_dict=datasets[labeled_split_name],
            data=labeled_dataset,
            train=True,
            grouper=grouper,
            config=config)
        configure_loaders(
            split_dict=datasets['unlabeled_test'],
            data=label_manager.get_unlabeled_subset(),
            train=False,
            grouper=None,
            config=config)

        # Then few-shot train on the new labels
        train(
            algorithm=few_shot_algorithm,
            datasets=datasets,
            train_split=labeled_split_name,
            val_split=None,
            unlabeled_split="unlabeled_test",
            general_logger=general_logger,
            config=config,
            rnd=rnd,
            epoch_offset=0,
            best_val_metric=None)

