from wilds.datasets.wilds_dataset import WILDSSubset
from torch.utils.data import Subset
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

def run_active_learning(algorithm, datasets, general_logger, config, epoch_offset, best_val_metric):
    import pdb
    pdb.set_trace()

    for round in range(config.n_rounds):
        general_logger.write('\nRound [%d]:\n' % round)

        # First run selection function
        label_manager = datasets['test']['label_manager']
        datasets['labeled_test']['dataset'], datasets['unlabeled_test']['dataset'] = random_sampling(label_manager, config.n_labels_round)

        few_shot_datasets = {
            'train': labeled_test,
            'val': labeled_test,
        }

        # Then run training
        train(
            algorithm=algorithm,
            datasets=few_shot_datasets,
            general_logger=general_logger,
            config=config,
            epoch_offset=epoch_offset,
            best_val_metric=best_val_metric)


#### SELECTION FUNCTIONS ####
def random_sampling(label_manager, n_labels_round):
    reveal = np.random.choice(
        list(label_manager.idx - label_manager.idx_labels_revealed),
        size=n_labels_round
    )
    label_manager.reveal_labels(reveal)
    return label_manager.get_labeled_subset(), label_manager.get_unlabeled_subset()
