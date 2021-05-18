from wilds.datasets.wilds_dataset import WILDSSubset
from torch.utils.data import Subset
from utils import configure_split_dict, configure_loaders
from train import train

class LabelManager:
    """
    Wraps a WILDS subset (e.g. ood test) with the ability to reveal / hide 
    labels and index these examples separately.
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

def run_active_learning(selection_fn, few_shot_algorithm, datasets, general_logger, grouper, config):
    label_manager = datasets['test']['label_manager']

    # Add labeled test / unlabeled test splits.
    datasets['labeled_test'] = configure_split_dict(
        data=None,
        split="labeled_test",
        split_name="labeled_test",
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
        selection_fn.select(label_manager=label_manager, K=config.n_labels_round)
        
        ## Refresh dataloaders
        configure_loaders(
            split_dict=datasets['labeled_test'],
            data=label_manager.get_labeled_subset(),
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
            train_split="labeled_test",
            val_split=None,
            general_logger=general_logger,
            config=config,
            rnd=rnd,
            epoch_offset=0,
            best_val_metric=None)
