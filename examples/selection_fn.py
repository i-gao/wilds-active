from wilds.common.data_loaders import get_train_loader, get_eval_loader

import numpy as np
import torch
import torch.nn.functional as F

def initialize_selection_function(config, uncertainty_model):
    # initialize selection function to choose target examples to label
    if config.selection_function=='random':
        selection_fn = RandomSampling()
    elif config.selection_function=='uncertainty':
        selection_fn = UncertaintySampling(uncertainty_model, config)
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
            size=K
        )
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
