import torch.nn as nn
import copy

def initialize_few_shot_algorithm(config, algorithm):
    """
    Modifies algorithm to be the Few Shot Learning algorithm
    Args:
        - algorithm: original algorithm class containing model trained on source
    """
    few_shot_algorithm = copy.deepcopy(algorithm)
    *_, last = few_shot_algorithm.modules()
    if config.few_shot_algorithm == "finetune":
        if config.few_shot_kwargs.get('reset_classifier'): 
            # re-initialize the last linear layer
            last.reset_parameters()
    elif config.few_shot_algorithm == "linear_probe":
        # freeze all 
        for param in few_shot_algorithm.model.parameters():
            param.requires_grad = False
        for param in last.parameters():
            param.requires_grad = True
        if config.few_shot_kwargs.get('reset_classifier'): 
            # re-initialize the last linear layer
            last.reset_parameters()
    else:
        raise ValueError(f'Selection Function {config.selection_function} not recognized.')
    # sanity check
    print(f"\nNumber of unfrozen parameters: {len(list(filter(lambda p: p.requires_grad, few_shot_algorithm.parameters())))}")
    return few_shot_algorithm 
