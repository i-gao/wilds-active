from wilds.common.data_loaders import get_train_loader, get_eval_loader

import numpy as np
import torch
import torch.nn.functional as F
import copy
from utils import configure_split_dict
from train import run_epoch
from tqdm import tqdm

def initialize_selection_function(config, orig_algorithm, few_shot_algorithm, grouper=None):
    # initialize selection function to choose target examples to label
    if config.selection_function=='random':
        selection_fn = RandomSampling()
    elif config.selection_function=='stratified':
        assert grouper is not None
        selection_fn = StratifiedSampling(grouper)
    elif config.selection_function=='uncertainty':
        if config.few_shot_kwargs.get('reset_classifier'): 
            import warnings
            warnings.warn("Running uncertainty sampling using a randomly initialized logits layer.")
        selection_fn = UncertaintySampling(few_shot_algorithm, config)
    elif config.selection_function=='uncertainty_fixed':
        selection_fn = UncertaintySampling(orig_algorithm, config)
    elif config.selection_function=='confidently_incorrect':
        selection_fn = ConfidentlyIncorrect(few_shot_algorithm, config)
    elif config.selection_function=='individual_oracle':
        selection_fn = IndividualOracle(few_shot_algorithm, grouper, config)
    else:
        raise ValueError(f'Selection Function {config.selection_function} not recognized.')
    return selection_fn

class SelectionFunction():
    """
    Abstract class for a function that selects K examples to reveal labels for.
    Works in conjunction with a LabelManager.
    """
    def __init__(self, is_trainable=False, config=None):
        """
        Args:
            - uncertainty_model: Algorithm
        """
        self.is_trainable = is_trainable
        self.config = config

    # def update(self):
    #     """
    #     Update the uncertainty model
    #     Args:
    #         - batch (tuple of Tensors): a batch of data yielded by data loaders
    #     """
    #     assert(self.is_trainable and self.uncertainty_model is not None, "This selection function does not use a trainable uncertainty model.")
    #     return self.uncertainty_model.update(batch)

    def select(self, label_manager, K):
        """
        Labels K points by passing label_manager the indexes of examples to reveal
        Args:
            - label_manager (LabelManager object): see active.py; keeps track of which test points have revealed labels
        """
        raise NotImplementedError

class RandomSampling(SelectionFunction):
    def __init__(self):
        super().__init__(
            is_trainable=False
        )
    
    def select(self, label_manager, K):
        reveal = np.random.choice(
            label_manager.unlabeled_indices,
            replace=False, # assuming this is large enough
            size=K
        )
        label_manager.reveal_labels(reveal)

class StratifiedSampling(SelectionFunction):
    def __init__(self, grouper):
        self.grouper = grouper
        super().__init__(
            is_trainable=False
        )
    
    def select(self, label_manager, K):
        groups, group_counts = self.grouper.metadata_to_group(
            label_manager.get_unlabeled_subset().metadata_array,
            return_counts=True)
        group_counts = group_counts.numpy().astype('float64')
        group_choices = np.random.choice(
            np.arange(len(group_counts)),
            K,
            p=group_counts/sum(group_counts))
        unlabeled_indices = np.array(label_manager.unlabeled_indices)
        reveal = [
            np.random.choice(
                unlabeled_indices[groups == g],
                size=sum(group_choices == g),
                replace=sum(group_choices == g) <= sum(groups == g))
            for g in range(len(group_counts))]
        reveal = np.concatenate(reveal).tolist()       
            
        label_manager.reveal_labels(reveal)

class UncertaintySampling(SelectionFunction):
    def __init__(self, uncertainty_model, config):
        self.uncertainty_model = uncertainty_model
        super().__init__(
            is_trainable=False,
            config=config
        )

    def select(self, label_manager, K):
        self.uncertainty_model.eval()
        # Get loader for estimating uncertainties
        unlabeled_test = label_manager.get_unlabeled_subset()
        loader = get_eval_loader(
            loader='standard',
            dataset=unlabeled_test,
            batch_size=self.config.batch_size,
            **self.config.loader_kwargs)

        # Get uncertainties
        certainties = []
        for batch in loader:
            res = self.uncertainty_model.evaluate(batch)
            logits = res['y_pred'] # before process_outputs_fn
            probs = F.softmax(logits, 1)
            certainties.append(torch.max(probs, 1)[0])
        certainties = torch.cat(certainties)

        # Choose K most uncertain to reval labels
        _, top_idxs = torch.topk(-certainties, K)
        reveal = torch.as_tensor(label_manager.unlabeled_indices)[top_idxs].tolist()
        label_manager.reveal_labels(reveal)

class ConfidentlyIncorrect(SelectionFunction):
    """oracle method: label the most confident incorrect predictions"""
    def __init__(self, uncertainty_model, config):
        self.uncertainty_model = uncertainty_model
        super().__init__(
            is_trainable=False,
            config=config
        )

    def select(self, label_manager, K):
        self.uncertainty_model.eval()
        # Get loader for estimating uncertainties
        unlabeled_test = label_manager.get_unlabeled_subset()
        loader = get_eval_loader(
            loader='standard',
            dataset=unlabeled_test,
            batch_size=self.config.batch_size,
            **self.config.loader_kwargs)

        # Get uncertainties and predictions
        certainties = []
        correct = []
        for x, y, m in loader:
            res = self.uncertainty_model.evaluate((x, y, m))
            logits = res['y_pred'] # before process_outputs_fn
            probs = F.softmax(logits, 1)
            certainty, pred = torch.max(probs, 1)
            certainties.append(certainty)
            correct.append(pred == y)
        certainties = torch.cat(certainties)
        correct = torch.cat(correct)

        # Choose K most certain and incorrect to reval labels
        _, top_idxs = torch.topk(certainties * ~correct, K)
        reveal = torch.as_tensor(label_manager.unlabeled_indices)[top_idxs].tolist()
        label_manager.reveal_labels(reveal)

class IndividualOracle(SelectionFunction):
    """oracle method: try a gradient step on every individual point & label those that best improve accuracy"""
    def __init__(self, uncertainty_model, grouper, config):
        self.uncertainty_model = uncertainty_model
        self.grouper = grouper
        super().__init__(
            is_trainable=False,
            config=config
        )

    def _get_delta_single_label(self, ind, label_manager):
        """Try an individual point and return change in val metric after one epoch"""
        label_manager.reveal_labels([ind])

        labeled_dict = configure_split_dict(
            data=label_manager.get_labeled_subset(),
            split="labeled_test",
            split_name="labeled_test",
            train=True,
            verbose=False,
            grouper=self.grouper,
            config=self.config)
        unlabeled_dict = configure_split_dict(
            data=label_manager.get_unlabeled_subset(),
            split="unlabeled_test",
            split_name="unlabeled_test",
            train=False,
            grouper=None,
            verbose=False,
            config=self.config)

        temp_model = copy.deepcopy(self.uncertainty_model)
        run_epoch( # train
            temp_model,
            labeled_dict,
            None,
            0,
            self.config,
            train=True)
        res, _ = run_epoch( # eval
            temp_model,
            unlabeled_dict,
            None,
            0,
            self.config,
            train=False)

        label_manager.hide_labels([ind])
        return res[self.config.val_metric]

    def select(self, label_manager, K):
        self.uncertainty_model.eval()
        unlabeled_indices = label_manager.unlabeled_indices

        label_manager.verbose = False
        delta = torch.zeros(len(unlabeled_indices), 1)
        for i, dataset_index in enumerate(tqdm(unlabeled_indices)):
            delta[i] = self._get_delta_single_label(dataset_index, label_manager)
        label_manager.verbose = True

        # Choose K improvement in val metric to reval labels
        if self.config.val_metric_decreasing: delta *= -1
        delta = delta.squeeze()

        _, top_idxs = torch.topk(delta, K)
        reveal = torch.as_tensor(label_manager.unlabeled_indices)[top_idxs].tolist()
        label_manager.reveal_labels(reveal)
