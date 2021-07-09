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

def initialize_selection_function(config, orig_algorithm, few_shot_algorithm, select_grouper, algo_grouper=None):
    # initialize selection function to choose target examples to label
    if config.selection_function=='random':
        selection_fn = RandomSampling(select_grouper, config)
    elif config.selection_function=='stratified':
        selection_fn = StratifiedSampling(select_grouper, config)
    elif config.selection_function=='uncertainty':
        if config.few_shot_kwargs.get('reset_classifier'): 
            import warnings
            warnings.warn("Running uncertainty sampling using a randomly initialized logits layer.")
        selection_fn = UncertaintySampling(few_shot_algorithm, select_grouper, config)
    elif config.selection_function=='uncertainty_fixed':
        selection_fn = UncertaintySampling(orig_algorithm, select_grouper, config)
    elif config.selection_function=='confidently_incorrect':
        selection_fn = ConfidentlyIncorrect(few_shot_algorithm, select_grouper, config)
    elif config.selection_function=='individual_oracle':
        selection_fn = IndividualOracle(few_shot_algorithm, select_grouper, config)
    elif config.selection_function=='approximate_individual_oracle':
        selection_fn = ApproximateIndividualOracle(few_shot_algorithm, select_grouper, config)
    elif config.selection_function=='approximate_group_oracle':
        selection_fn = ApproximateGroupOracle(few_shot_algorithm, select_grouper, config)
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
    def __init__(self, select_grouper, is_trainable=False, config=None):
        """
        Args:
            - is_trainable: selection_fn depends on an uncertainty_model that is separately trained
        """
        self.select_grouper = select_grouper
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
        Labels K examples by passing label_manager the indexes of examples to reveal
        Wrapper for the select() function implemented by subclasses
        """
        groups = self.select_grouper.metadata_to_group(label_manager.unlabeled_metadata_array)
        group_ids = groups.unique().int().tolist()
        remaining = (torch.ones(len(group_ids)) * K).int().tolist()
        reveal = []
        for idx in self._prior_selections:
            i = label_manager.unlabeled_indices.index(idx)
            g = groups[i]
            g_ind = group_ids.index(g)
            if remaining[g_ind] > 0:
                reveal.append(idx)
                remaining[g_ind] -= 1
            if sum(remaining) == 0: break
        self._prior_selections = [idx for idx in self._prior_selections if idx not in set(reveal)]

        if sum(remaining) > 0:
            unlabeled_indices = torch.tensor(label_manager.unlabeled_indices)
            reveal = reveal + self.select(label_manager, remaining, unlabeled_indices, groups, group_ids)

        label_manager.reveal_labels(reveal)
        self.save_selections(reveal)

    def select(self, label_manager, K_per_group:[int], unlabeled_indices: torch.Tensor, groups: torch.Tensor, group_ids:[int]):
        """
        Selects examples from the label manager's unlabeled subset to label.
        Abstract fn implemented in each child class
        Args:
            - label_manager (LabelManager object): see active.py; keeps track of which test points have revealed labels
            - K_per_group: list where K_per_group[i] is the number of examples left to label in group group_ids[i]
            - unlabeled_indices: (torch Tensor) output of label_manager.unlabeled_indices, cast as a tensor
            - groups: (torch Tensor) groups[i] is the group of example label_manager.unlabeled_indices[i]
            - group_ids: names of groups; shares indexing with K_per_group
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
    def __init__(self, select_grouper, config):
        super().__init__(
            select_grouper=select_grouper,
            is_trainable=False,
            config=config
        )
    
    def select(self, label_manager, K_per_group, unlabeled_indices, groups, group_ids):
        reveal = []
        for i, K in enumerate(K_per_group):
            g = group_ids[i]
            reveal_g = np.random.choice(
                unlabeled_indices[groups == g],
                replace=False, # assuming this is large enough
                size=min(K, sum(groups == g))
            ).tolist()
            reveal = reveal + reveal_g
        return reveal

class UncertaintySampling(SelectionFunction):
    def __init__(self, uncertainty_model, select_grouper, config):
        self.uncertainty_model = uncertainty_model
        super().__init__(
            select_grouper=select_grouper,
            is_trainable=False,
            config=config
        )

    def select(self, label_manager, K_per_group, unlabeled_indices, groups, group_ids):
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

        # Choose most uncertain to reveal labels
        reveal = []
        for i, K in enumerate(K_per_group):
            g = group_ids[i]
            _, top_idxs = torch.topk(-certainties[groups == g], K)
            reveal_g = unlabeled_indices[groups == g][top_idxs].tolist()
            reveal = reveal + reveal_g
        return reveal

class ConfidentlyIncorrect(SelectionFunction):
    """oracle method: label the most confident incorrect predictions"""
    def __init__(self, uncertainty_model, select_grouper, config):
        self.uncertainty_model = uncertainty_model
        super().__init__(
            select_grouper=select_grouper,
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

        # Choose most certain and incorrect to reveal labels
        _, top_idxs = torch.topk(certainties * ~correct, K)
        reveal = torch.as_tensor(label_manager.unlabeled_indices)[top_idxs].tolist()
        return reveal

class Oracle(SelectionFunction):
    """oracle method: try a gradient step on some points & label those that best improve accuracy"""
    def __init__(self, uncertainty_model, algo_grouper, select_grouper, config):
        self.uncertainty_model = uncertainty_model
        self.algo_grouper = algo_grouper
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

        # Choose improvement in val metric to reveal labels
        if self.config.val_metric_decreasing: delta *= -1
        _, top_idxs = torch.topk(delta, K)
        reveal = torch.as_tensor(label_manager.unlabeled_indices)[top_idxs].tolist()
        return reveal

class ApproximateIndividualOracle(Oracle):
    """oracle method: try a gradient step on G randomly sampled individual points & label those that best improve accuracy"""
    def __init__(self, uncertainty_model, select_grouper, config):
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

        # Choose improvement in val metric to reveal labels
        if self.config.val_metric_decreasing: delta *= -1
        _, top_idxs = torch.topk(delta, K)
        reveal = torch.as_tensor(sampled_unlabeled_indices)[top_idxs].tolist()
        return reveal

class ApproximateGroupOracle(Oracle):
    """oracle method: try a gradient step on G randomly sampled groups of K & label group that best improves accuracy"""
    def __init__(self, uncertainty_model, select_grouper, config):
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

        # Choose improvement in val metric to reveal labels
        if self.config.val_metric_decreasing: delta *= -1
        top_idx = torch.argmax(delta)
        reveal = sampled_unlabeled_indices[top_idx].tolist()
        return reveal
