from wilds.datasets.wilds_dataset import WILDSSubset
from torch.utils.data import Subset

def initialize_selection_function(config, model):
    # initialize selection function to choose target examples to label
    if config.selection_function=='random':
        params = filter(lambda p: p.requires_grad, model.parameters())
        selection_fn = None
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

