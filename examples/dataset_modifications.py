"""
Functions that modify specific WILDSDatasets from the WILDS package due to our labeled data coming from the test split setup.
"""
from wilds.datasets.wilds_dataset import WILDSSubset
import torch
import numpy as np
from utils import PseudolabeledSubset

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

    def hide_labels(self, idx: [int]):
        """Unremembers revealed example labels, undoing reveal_labels()"""
        if not (set(idx).issubset(self._idx)):
            raise ValueError('Some indices are invalid.')
        if not (set(idx).issubset(self._idx_labels_revealed)):
            raise ValueError('Some indices were never revealed.')
        self._idx_labels_revealed = self._idx_labels_revealed - set(idx)

    @property
    def num_labeled(self):
        return len(self._idx_labels_revealed)
    
    @property
    def num_unlabeled(self):
        return len(self._idx - self._idx_labels_revealed)

    @property
    def labeled_indices(self):
        return sorted(list(self._idx_labels_revealed))

    @property
    def unlabeled_indices(self):
        return sorted(list(self._idx - self._idx_labels_revealed))
    
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


def add_split_to_wilds_dataset_metadata_array(full_dataset):
    """
    In this project, we sometimes train on batches of mixed splits, e.g. some train labeled examples and test labeled examples
    Within each batch, we may want to sample uniformly across split, or log the train v. test label balance
    To facilitate this, we'll hack the WILDS dataset to include each point's split in the metadata array
    """
    UNUSED_SPLIT = 10 # this doesn't overlap with any real split values in WILDS
    split_array = torch.tensor(full_dataset.split_array).unsqueeze(1) 
    split_array[split_array < 0] = UNUSED_SPLIT # unused data is given a split of -1, but the grouper can only accept nonnegative values
    full_dataset._metadata_array = torch.cat((
        full_dataset.metadata_array, 
        split_array,
    ), dim=1) # add split as a metadata column
    full_dataset._metadata_fields.append('split')