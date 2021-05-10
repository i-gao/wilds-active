import torch.nn as nn
import copy
from algorithms.initializer import initialize_algorithm

def initialize_few_shot_algorithm(config, algorithm):
    """
    Modifies algorithm to be the Few Shot Learning algorithm
    Args:
        - algorithm: original algorithm class containing model trained on source
    """
    if config.few_shot_algorithm == "finetune":
        few_shot_algorithm = copy.deepcopy(algorithm)
        if config.few_shot_kwargs.get('reset_classifier'): 
            # re-initialize the last linear layer
            few_shot_algorithm.model.classifier = nn.Linear(
                algorithm.model.classifier.in_features, 
                algorithm.model.classifier.out_features, 
                bias=True)
    else:
        raise ValueError(f'Selection Function {config.selection_function} not recognized.')
    return few_shot_algorithm 
