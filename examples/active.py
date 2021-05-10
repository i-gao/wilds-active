from wilds.datasets.wilds_dataset import WILDSSubset
from torch.utils.data import Subset
from utils import configure_split_dict
from train import train

class LabelManager:
    """
    Wraps a WILDS subset (e.g. ood test) with the ability to reveal / hide 
    labels and index these examples separately.
    """
    def __init__(self, subset: WILDSSubset):
        self.dataset = subset.dataset
        self.transform = subset.transform
        self.idx = set(subset.indices)
        self.idx_labels_revealed = set()

    def get_unlabeled_subset(self):
        return WILDSSubset(self.dataset, list(self.idx - self.idx_labels_revealed), self.transform)

    def get_labeled_subset(self):
        return WILDSSubset(self.dataset, list(self.idx_labels_revealed), self.transform)

    def reveal_labels(self, idx: [int]):
        """Remembers these examples as having labels revealed, 
        so these examples will be returned via a different data loader"""
        if not (set(idx).issubset(self.idx)):
            raise ValueError(f'Some indices are invalid.')
        self.idx_labels_revealed.update(idx)
        print(f"Total Labels Revealed: {len(self.idx_labels_revealed)}")

def run_active_learning(selection_fn, few_shot_algorithm, datasets, general_logger, grouper, config):
    label_manager = datasets['test']['label_manager']
    for round in range(config.n_rounds):
        general_logger.write('\nActive Learning Round [%d]:\n' % round)
        
        # First run selection function
        ## Train selection function
        # selection_fn.update()
        ## Get a few labels
        selection_fn.select(label_manager=label_manager, K=config.n_labels_round)
        
        ## Set up dataloaders. Must be reloaded each time more labels are revealed.
        datasets['labeled_test'] = configure_split_dict(
            data=label_manager.get_labeled_subset(),
            split="test",
            split_name="labeled_test",
            train=True,
            verbose=True,
            grouper=grouper,
            config=config)
        datasets['unlabeled_test'] = configure_split_dict(
            data=label_manager.get_unlabeled_subset(),
            split="test",
            split_name="unlabeled_test",
            train=False,
            grouper=None,
            verbose=True,
            config=config)

        # Then few-shot train on the new labels
        train(
            algorithm=few_shot_algorithm,
            datasets=datasets,
            train_split="labeled_test",
            val_split=None,
            general_logger=general_logger,
            config=config,
            epoch_offset=0,
            best_val_metric=None)