import torch.nn as nn
import torch
import sys, os

# metrics
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from wilds.common.metrics.all_metrics import Accuracy, MultiTaskAccuracy, MSE, multiclass_logits_to_pred, binary_logits_to_pred
from utils import cross_entropy_with_logits_loss

losses = {
    'cross_entropy': ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none')),
    'cross_entropy_logits': ElementwiseLoss(loss_fn=cross_entropy_with_logits_loss),
    'lm_cross_entropy': MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction='none')),
    'mse': MSE(name='loss'),
    'multitask_bce': MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none')),
}

algo_log_metrics = {
    'accuracy': Accuracy(prediction_fn=multiclass_logits_to_pred),
    'mse': MSE(),
    'multitask_accuracy': MultiTaskAccuracy(prediction_fn=multiclass_logits_to_pred),
    'multitask_binary_accuracy': MultiTaskAccuracy(prediction_fn=binary_logits_to_pred),
    None: None,
}

process_outputs_functions = {
    'binary_logits_to_pred': binary_logits_to_pred,
    'multiclass_logits_to_pred': multiclass_logits_to_pred,
    None: None,
}

# see initialize_*() functions for correspondence
transforms = ['bert', 'image_base', 'image_resize_and_center_crop', 'poverty_train']
models = ['resnet18_ms', 'resnet50', 'resnet34', 'wideresnet50',
         'densenet121', 'bert-base-uncased', 'distilbert-base-uncased',
         'gin-virtual', 'logistic_regression', 'code-gpt-py']
algorithms = ['ERM', 'groupDRO', 'deepCORAL', 'IRM', 'MAML', 'ANIL', 'FixMatch', 'PseudoLabel', 'NoisyStudent']
optimizers = ['SGD', 'Adam', 'AdamW']
schedulers = ['linear_schedule_with_warmup', 'ReduceLROnPlateau', 'StepLR']
selection_functions = [
    'random', 'uncertainty', 'uncertainty_fixed', 
    'highest_loss', 'confidently_incorrect', 'approximate_lookahead'
]
few_shot_algorithms = ['finetune', 'linear_probe']