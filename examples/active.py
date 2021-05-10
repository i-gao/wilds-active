from wilds.datasets.wilds_dataset import WILDSSubset
from torch.utils.data import Subset
from utils import configure_split_dict
from train import train
import numpy as np

def initialize_selection_function(config, model):
    # initialize selection function to choose target examples to label
    if config.selection_function=='random':
        params = filter(lambda p: p.requires_grad, model.parameters())
        selection_fn = random_sampling
    else:
        raise ValueError(f'Selection Function {config.selection_function} not recognized.')
    return selection_fn

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

def run_active_learning(algorithm, datasets, general_logger, grouper, config):
    import pdb
    pdb.set_trace()

    for round in range(config.n_rounds):
        general_logger.write('\nRound [%d]:\n' % round)

        # First run selection function
        labeled_test, unlabeled_test = random_sampling(datasets['test']['label_manager'], config.n_labels_round)
        datasets['labeled_test'] = configure_split_dict(
            data=labeled_test,
            split="test",
            split_name="labeled_test",
            train=True,
            verbose=True,
            grouper=grouper,
            config=config)
        datasets['unlabeled_test'] = configure_split_dict(
            data=unlabeled_test,
            split="test",
            split_name="unlabeled_test",
            train=False,
            grouper=None,
            verbose=True,
            config=config)

        # Then run training
        train(
            algorithm=algorithm,
            datasets=few_shot_datasets,
            general_logger=general_logger,
            config=config,
            epoch_offset=0,
            best_val_metric=None)


#### SELECTION FUNCTIONS ####
def random_sampling(label_manager, n_labels_round):
    reveal = np.random.choice(
        list(label_manager.idx - label_manager.idx_labels_revealed),
        size=n_labels_round
    )
    label_manager.reveal_labels(reveal)
    return label_manager.get_labeled_subset(), label_manager.get_unlabeled_subset()
