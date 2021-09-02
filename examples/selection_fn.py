from wilds.common.data_loaders import get_train_loader, get_eval_loader

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import copy
from utils import configure_split_dict
from train import run_epoch
from tqdm import tqdm
import glob
from shutil import copyfile

def initialize_selection_function(config, algorithm, select_grouper, algo_grouper=None):
    # initialize selection function to choose target examples to label
    if config.selection_function=='random':
        selection_fn = RandomSampling(select_grouper, config)
    elif config.selection_function=='uncertainty':
        selection_fn = UncertaintySampling(algorithm, select_grouper, config)
    elif config.selection_function=='uncertainty_fixed':
        selection_fn = UncertaintySampling(copy.deepcopy(algorithm), select_grouper, config)
    elif config.selection_function=='highest_loss':
        selection_fn = HighestLoss(algorithm, select_grouper, config)
    elif config.selection_function=='max_grad_norm':
        selection_fn = MaxGradientNorm(algorithm, select_grouper, config)
    elif config.selection_function=='approximate_lookahead':
        selection_fn = ApproximateLookahead(algorithm, algo_grouper, select_grouper, config)
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
        if K == 0: return
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

    def load_selections(self, path):
        """
        Loads indices to select in the order saved in given csv
        """
        if path.endswith('.csv'): csvpath = path
        else: csvpath = f'{path}/selections.csv'

        try: df = pd.read_csv(csvpath, index_col=None, header=None)
        except:
            print(f"Couldn't find this file of previous selections: {csvpath}.")
            
        assert len(df.columns) == 1
        self._prior_selections.append(df[0].tolist())
        
        print(f"Loaded {len(self._prior_selections)} previous selections")

    def save_selections(self, indices):
        """
        Saves indices of selected points
        """
        csvpath = f"{self.log_dir}/selections.csv"
        df = pd.DataFrame(indices)
        df.to_csv(csvpath, mode=self.mode, index=False, header=False)

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
            K = min(K, sum(groups == g).int().item())
            reveal_g = np.random.choice(
                unlabeled_indices[groups == g],
                replace=False, # assuming this is large enough
                size=K
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
        loader = get_eval_loader(
            loader='standard',
            dataset=label_manager.get_unlabeled_subset(), # TODO: should this be augmented or unaugmented examples?
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
            K = min(K, sum(groups == g).int().item())
            _, top_idxs = torch.topk(-certainties[groups == g], K)
            reveal_g = unlabeled_indices[groups == g][top_idxs].tolist()
            reveal = reveal + reveal_g
        return reveal

######### ORACLES (use label information) #########

class HighestLoss(SelectionFunction):
    """oracle method: label the examples with highest loss, as defined by the algorithm
    i.e. ERM will use highest classification loss, PseudoLabel will use highest classification + consistency loss.
    """
    def __init__(self, uncertainty_model, select_grouper, config):
        self.uncertainty_model = uncertainty_model
        super().__init__(
            select_grouper=select_grouper,
            is_trainable=False,
            config=config
        )

    def select(self, label_manager, K_per_group, unlabeled_indices, groups, group_ids):
        self.uncertainty_model.eval()
        # Get loader for estimating loss
        loader = get_eval_loader(
            loader='standard',
            dataset=label_manager.get_unlabeled_subset(),
            batch_size=1,
            **self.config.loader_kwargs)
        iterator = tqdm(loader) if self.config.progress_bar else loader

        # Get losses
        losses = []
        for batch in iterator:
            # pass as both labeled and unlabeled to get classification loss and consistency loss
            res = self.uncertainty_model.evaluate(batch, unlabeled_batch=batch)
            losses.append(res['objective']) # float, not torch float
        losses = torch.tensor(losses)
        
        # Choose highest loss to reveal labels
        reveal = []
        for i, K in enumerate(K_per_group):
            g = group_ids[i]
            K = min(K, sum(groups == g).int().item())
            _, top_idxs = torch.topk(losses[groups == g], K)
            reveal_g = unlabeled_indices[groups == g][top_idxs].tolist()
            reveal = reveal + reveal_g
        return reveal

class MaxGradientNorm(SelectionFunction):
    """oracle method: label the examples with the highest loss gradient norm, where loss is defined by the algorithm
    i.e. ERM will use grad of classification loss, PseudoLabel will use grad of classification + consistency loss.
    Approximation of retraining-based methods.
    """
    def __init__(self, uncertainty_model, select_grouper, config):
        self.uncertainty_model = uncertainty_model
        super().__init__(
            select_grouper=select_grouper,
            is_trainable=False,
            config=config
        )

    def select(self, label_manager, K_per_group, unlabeled_indices, groups, group_ids):
        self.uncertainty_model.train()
        # Get loader for estimating gradients
        loader = get_eval_loader(
            loader='standard',
            dataset=label_manager.get_unlabeled_subset(),
            batch_size=1,
            **self.config.loader_kwargs)
        iterator = tqdm(loader) if self.config.progress_bar else loader

        # Get gradients
        grads = []
        for batch in iterator:
            x = batch[0]
            x.requires_grad = True
            res = self.uncertainty_model.process_batch(batch, unlabeled_batch=batch)
            obj = self.uncertainty_model.objective(res)
            obj.backward()
            norm = torch.linalg.norm(x.grad)
            grads.append(norm.item()) # float, not torch float
            self.uncertainty_model.sanitize_dict(res)
        grads = torch.tensor(grads)
        
        # Choose max gradient norm to reveal labels
        reveal = []
        for i, K in enumerate(K_per_group):
            g = group_ids[i]
            K = min(K, sum(groups == g).int().item())
            _, top_idxs = torch.topk(grads[groups == g], K)
            reveal_g = unlabeled_indices[groups == g][top_idxs].tolist()
            reveal = reveal + reveal_g
        return reveal

class MStepLookahead(SelectionFunction):
    """oracle method: retraining-based; take M gradients step on some points & label those that best improve accuracy"""
    def __init__(self, uncertainty_model, algo_grouper, select_grouper, config):
        self.uncertainty_model = uncertainty_model
        self.grouper = algo_grouper
        self.M = config.selection_function_kwargs.get('n_steps', 1)
        super().__init__(
            select_grouper=select_grouper,
            is_trainable=False,
            config=config
        )

    def _get_delta(self, indices, label_manager):
        """Try adding indices to the labeled set, train on labeled set, and return change in val metric after M epochs"""
        if type(indices) != list: indices = [indices]
        label_manager.reveal_labels(indices)

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
        for epoch in range(self.M):
            run_epoch(temp_model, labeled_dict, None, epoch, self.config, train=True)
        res, _ = run_epoch(temp_model, unlabeled_dict, None, 0, self.config, train=False)
        del temp_model

        label_manager.hide_labels(indices)
        return res[self.config.val_metric]

class ApproximateLookahead(MStepLookahead):
    """oracle method: try M gradient steps on G randomly sampled groups of K & label group that best improves accuracy"""
    def __init__(self, uncertainty_model, algo_grouper, select_grouper, config):
        self.G = config.selection_function_kwargs.get('n_simulations', 100)
        self.random_sampler = RandomSampling(select_grouper, config)
        super().__init__(
            uncertainty_model=uncertainty_model,
            algo_grouper=algo_grouper,
            select_grouper=select_grouper,
            config=config
        )
    
    def select(self, label_manager, K_per_group, unlabeled_indices, groups, group_ids):
        samples = [self.random_sampler.select(label_manager, K_per_group, unlabeled_indices, groups, group_ids) for _ in range(self.G)]
        iterator = tqdm(samples) if self.config.progress_bar else samples

        label_manager.verbose = False
        delta = torch.zeros(len(samples))
        for i, sample in enumerate(iterator):
            delta[i] = self._get_delta(sample, label_manager)
        label_manager.verbose = True

        # Choose improvement in val metric to reveal labels
        if self.config.val_metric_decreasing: delta *= -1
        top_idx = torch.argmax(delta).item()
        reveal = samples[top_idx]
        return reveal
