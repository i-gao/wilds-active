from wilds.common.data_loaders import get_train_loader, get_eval_loader

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import copy
from utils import configure_split_dict
from train import run_epoch
from tqdm import tqdm
import os
from shutil import copyfile

def initialize_selection_function(config, orig_algorithm, few_shot_algorithm, grouper=None):
    # initialize selection function to choose target examples to label
    if config.selection_function=='random':
        selection_fn = RandomSampling(config)
    elif config.selection_function=='stratified':
        assert grouper is not None
        selection_fn = StratifiedSampling(grouper, config)
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
    elif config.selection_function=='approximate_individual_oracle':
        selection_fn = ApproximateIndividualOracle(few_shot_algorithm, grouper, config)
    elif config.selection_function=='approximate_group_oracle':
        selection_fn = ApproximateGroupOracle(few_shot_algorithm, grouper, config)
    else:
        raise ValueError(f'Selection Function {config.selection_function} not recognized.')

    if config.selection_function_kwargs.get('load_selection_path'):
        selection_fn.load_selections(config.selection_function_kwargs['load_selection_path'])

    return selection_fn

class SelectionFunction():
    """
    Abstract class for a function that selects K examples to reveal labels for.
    Works in conjunction with a LabelManager.
    """
    def __init__(self, is_trainable=False, config=None):
        """
        Args:
            - is_trainable: selection_fn depends on an uncertainty_model that is separately trained
        """
        self.is_trainable = is_trainable
        self.config = config
        self._prior_selections = [] # loaded selections from file
        self.log_dir = self.config.log_dir
        self.mode = 'w'

        # for safety, copy previous selections.csv -> selections_old.csv
        if os.path.exists(f"{self.log_dir}/selections.csv"): 
            copyfile(f"{self.log_dir}/selections.csv", f"{self.log_dir}/selections_old.csv")

    # def update(self):
    #     """
    #     Update the uncertainty model
    #     Args:
    #         - batch (tuple of Tensors): a batch of data yielded by data loaders
    #     """
    #     assert(self.is_trainable and self.uncertainty_model is not None, "This selection function does not use a trainable uncertainty model.")
    #     return self.uncertainty_model.update(batch)

    def select_and_reveal(self, label_manager, K):
        """
        Labels K points by passing label_manager the indexes of examples to reveal
        Wrapper for the select() function implemented by individual subclasse
        """
        reveal = []
        if len(self._prior_selections):
             reveal = self._prior_selections[:K]
             del self._prior_selections[:K]

        remaining_K = K - len(reveal)
        if remaining_K:
             reveal = reveal + self.select(label_manager, remaining_K)

        label_manager.reveal_labels(reveal)
        self.save_selections(reveal)

    def select(self, label_manager, K):
        """
        Labels K points by passing label_manager the indexes of examples to reveal
        Abstract fn implemented in each child class
        Args:
            - label_manager (LabelManager object): see active.py; keeps track of which test points have revealed labels
        """
        raise NotImplementedError

    def load_selections(self, csvpath):
        """
        Loads indices to select in the order of some csv
        """
        try:
            df = pd.read_csv(csvpath, index_col=None, header=None)
        except:
            print(f"Couldn't find this file of previous selections: {csvpath}.")
            return 
        
        assert len(df.columns) == 1
        self._prior_selections = df[0].tolist()
        print(f"Loaded {len(self._prior_selections)} previous selections")

    def save_selections(self, indices):
        """
        Saves indices of selected points
        """
        csvpath = f"{self.log_dir}/selections.csv"
        df = pd.DataFrame(indices)
        df.to_csv(csvpath, mode=self.mode, index=False, header=False)
        # now that we've written, append future rounds
        self.mode = 'a'

class RandomSampling(SelectionFunction):
    def __init__(self, config):
        super().__init__(
            is_trainable=False,
            config=config
        )
    
    def select(self, label_manager, K):
        reveal = np.random.choice(
            label_manager.unlabeled_indices,
            replace=False, # assuming this is large enough
            size=K
        ).tolist()
        return reveal
class StratifiedSampling(SelectionFunction):
    def __init__(self, grouper, config):
        self.grouper = grouper
        super().__init__(
            is_trainable=False,
            config=config
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
        return reveal

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
        return reveal

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
        return reveal

class Oracle(SelectionFunction):
    """oracle method: try a gradient step on some points & label those that best improve accuracy"""
    def __init__(self, uncertainty_model, grouper, config):
        self.uncertainty_model = uncertainty_model
        self.grouper = grouper
        super().__init__(
            is_trainable=False,
            config=config
        )

    def _get_delta(self, ind, label_manager):
        """Try an individual point and return change in val metric after one epoch"""
        if type(ind) != list: ind = [ind]
        label_manager.reveal_labels(ind)

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
        temp_model.train()
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
        del temp_model

        label_manager.hide_labels(ind)
        return res[self.config.val_metric]

    def select(self, label_manager, K):
        pass

class IndividualOracle(Oracle):
    """oracle method: try a gradient step on all individual points & label those that best improve accuracy"""
    def select(self, label_manager, K):
        unlabeled_indices = label_manager.unlabeled_indices

        label_manager.verbose = False
        delta = torch.zeros(len(unlabeled_indices))
        for i, dataset_index in enumerate(tqdm(unlabeled_indices)):
            delta[i] = self._get_delta(dataset_index, label_manager)
        label_manager.verbose = True

        # Choose K improvement in val metric to reval labels
        if self.config.val_metric_decreasing: delta *= -1
        _, top_idxs = torch.topk(delta, K)
        reveal = torch.as_tensor(label_manager.unlabeled_indices)[top_idxs].tolist()
        return reveal

class ApproximateIndividualOracle(Oracle):
    """oracle method: try a gradient step on G randomly sampled individual points & label those that best improve accuracy"""
    def __init__(self, uncertainty_model, grouper, config):
        self.G = config.selection_function_kwargs.get('n_simulations', 100)
        super().__init__(
            uncertainty_model=uncertainty_model,
            grouper=grouper,
            config=config
        )
    
    def select(self, label_manager, K):
        assert self.G > K
        unlabeled_indices = label_manager.unlabeled_indices
        sampled_unlabeled_indices = np.random.choice(unlabeled_indices, size=min(len(unlabeled_indices), self.G), replace=False)

        label_manager.verbose = False
        delta = torch.zeros(len(sampled_unlabeled_indices))
        for i, dataset_index in enumerate(tqdm(sampled_unlabeled_indices)):
            delta[i] = self._get_delta(dataset_index, label_manager)
        label_manager.verbose = True

        # Choose K improvement in val metric to reval labels
        if self.config.val_metric_decreasing: delta *= -1
        _, top_idxs = torch.topk(delta, K)
        reveal = torch.as_tensor(sampled_unlabeled_indices)[top_idxs].tolist()
        return reveal

class ApproximateGroupOracle(Oracle):
    """oracle method: try a gradient step on G randomly sampled groups of K & label group that best improves accuracy"""
    def __init__(self, uncertainty_model, grouper, config):
        self.G = config.selection_function_kwargs.get('n_simulations', 100)
        super().__init__(
            uncertainty_model=uncertainty_model,
            grouper=grouper,
            config=config
        )
    
    def select(self, label_manager, K):
        unlabeled_indices = label_manager.unlabeled_indices
        sampled_unlabeled_indices = [np.random.choice(unlabeled_indices, size=K, replace=False) for _ in range(self.G)]

        label_manager.verbose = False
        delta = torch.zeros(len(sampled_unlabeled_indices))
        for i, dataset_indices in enumerate(tqdm(sampled_unlabeled_indices)):
            delta[i] = self._get_delta(dataset_indices.tolist(), label_manager)
        label_manager.verbose = True

        # Choose K improvement in val metric to reval labels
        if self.config.val_metric_decreasing: delta *= -1
        top_idx = torch.argmax(delta)
        reveal = sampled_unlabeled_indices[top_idx].tolist()
        return reveal
