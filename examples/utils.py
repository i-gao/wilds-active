import sys
import os
import csv
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import re

from algorithms.algorithm import Algorithm

from collections import defaultdict
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.wilds_dataset import WILDSSubset

try:
    import wandb
except Exception as e:
    pass

class WILDSPseudolabeledSubset(WILDSSubset):
    """Pseudolabeled subset initialized from a labeled subset"""
    def __init__(self, reference_subset, pseudolabels, transform):
        assert len(reference_subset) == len(pseudolabels)
        self.pseudolabels = pseudolabels
        super().__init__(
            reference_subset.dataset, reference_subset.indices, transform
        )

    def __getitem__(self, idx):
        import pdb
        pdb.set_trace()
        x, y_true, metadata = self.dataset[self.indices[idx]]
        y_pseudo = self.pseudolabels[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y_pseudo, y_true, metadata

def cross_entropy_with_logits_loss(input, soft_target):
    """
    Implementation of CrossEntropy loss using a soft target. Extension of BCEWithLogitsLoss to MCE.
    Normally, cross entropy loss is 
        \sum_j 1{j == y} -log \frac{e^{s_j}}{\sum_k e^{s_k}} = -log \frac{e^{s_y}}{\sum_k e^{s_k}}
    Here we use
        \sum_j P_j *-log \frac{e^{s_j}}{\sum_k e^{s_k}}
    where 0 <= P_j <= 1    
    Does not support fancy nn.CrossEntropy options (e.g. weight, size_average, ignore_index, reductions, etc.)
    
    Args:
    - input (N, k): logits
    - soft_target (N, k): targets for softmax(input); likely want to use class probabilities
    Returns:
    - losses (N, 1)
    """
    return torch.sum(- soft_target * torch.nn.functional.log_softmax(input, 1), 1)

def update_average(prev_avg, prev_counts, curr_avg, curr_counts):
    denom = prev_counts + curr_counts
    if isinstance(curr_counts, torch.Tensor):
        denom += (denom==0).float()
    elif isinstance(curr_counts, int) or isinstance(curr_counts, float):
        if denom==0:
            return 0.
    else:
        raise ValueError('Type of curr_counts not recognized')
    prev_weight = prev_counts/denom
    curr_weight = curr_counts/denom
    return prev_weight*prev_avg + curr_weight*curr_avg

# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-','').isnumeric():
                processed_val = int(value_str)
            elif value_str.replace('-','').replace('.','').isnumeric():
                processed_val = float(value_str)
            elif value_str in ['True', 'true']:
                processed_val = True
            elif value_str in ['False', 'false']:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val

def parse_bool(v):
    if v.lower()=='true':
        return True
    elif v.lower()=='false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_model(algorithm, epoch, best_val_metric, path):
    state = {}
    state['algorithm'] = algorithm.state_dict()
    state['epoch'] = epoch
    state['best_val_metric'] = best_val_metric
    torch.save(state, path)

def freeze_features(algorithm):
    """Freezes all params except the last layer"""
    *_, last = algorithm.modules()
    for param in algorithm.model.parameters():
        param.requires_grad = False
    for param in last.parameters():
        param.requires_grad = True
    print(f"\nNumber of unfrozen parameters: {len(list(filter(lambda p: p.requires_grad, algorithm.parameters())))}")

def load(module, path, device=None, tries=2):
    """Handles loading weights saved from this repo/model into an algorithm/model."""
    if device is not None:
        state = torch.load(path, map_location=device)
    else:
        state = torch.load(path)

    ## edge case: if module is a metalearning model, we want to load the state into self.meta_model.module (l2l artifact)
    if hasattr(module, 'meta_model'): module = module.meta_model.module

    if 'algorithm' in state:
        prev_epoch = state['epoch']
        best_val_metric = state['best_val_metric']
        state = state['algorithm']
    else:
        prev_epoch, best_val_metric = None, None

    # naive approach works if alg -> alg / mod -> mod
    try: module.load_state_dict(state)
    except:
        module_keys = module.state_dict().keys()
        for _ in range(tries):
            state = match_keys(state, list(module_keys))
            module.load_state_dict(state, strict=False)
            leftover_state = {k:v for k,v in state.items() if k in list(state.keys()-module_keys)}
            leftover_module_keys = module_keys - state.keys()
            if len(leftover_state) == 0 or len(leftover_module_keys) == 0: break
            state, module_keys = leftover_state, leftover_module_keys
        if len(module_keys-state.keys()) > 0: print(f"Some module parameters could not be found in the loaded state: {module_keys-state.keys()}")
    return prev_epoch, best_val_metric

def match_keys(d, ref):
    """
    Matches the format of keys between d (a dict) and ref (a list of keys).

    Helper function for situations where two algorithms share the same model, and we'd like to warm-start one
    algorithm with the model of another. Some algorithms (e.g. FixMatch) save the featurizer, classifier within a sequential,
    and thus the featurizer keys may look like 'model.module.0._' 'model.0._' or 'model.module.model.0._',
    and the classifier keys may look like 'model.module.1._' 'model.1._' or 'model.module.model.1._'
    while simple algorithms (e.g. ERM) use no sequential 'model._'
    """
    # hard-coded exceptions
    d = {re.sub('model.1.', 'model.classifier.', k): v for k,v in d.items()}
    d = {k: v for k,v in d.items() if 'pre_classifier' not in k} # this causes errors

    # probe the proper transformation from d.keys() -> reference
    # do this by splitting d's first key on '.' until we get a string that is a strict substring of something in ref
    success = False
    probe = list(d.keys())[0].split('.')
    for i in range(len(probe)):
        probe_str = '.'.join(probe[i:])
        matches = list(filter(lambda ref_k: len(ref_k) >= len(probe_str) and probe_str == ref_k[-len(probe_str):], ref))
        matches = list(filter(lambda ref_k: not 'layer' in ref_k, matches)) # handle resnet probe being too simple, e.g. 'weight'
        if len(matches) == 0: continue
        else:
            success = True
            append = [m[:-len(probe_str)] for m in matches]
            remove = '.'.join(probe[:i]) + '.'
            break
    if not success: raise Exception("These dictionaries have irreconcilable keys")

    return_d = {}
    for a in append:
        for k,v in d.items(): return_d[re.sub(remove, a, k)] = v

    # hard-coded exceptions
    if 'model.classifier.weight' in return_d:
       return_d['model.1.weight'], return_d['model.1.bias'] = return_d['model.classifier.weight'], return_d['model.classifier.bias']
    return return_d

def log_group_data(datasets, grouper, logger):
    for k, dataset in datasets.items():
        name = dataset['name']
        dataset = dataset['dataset']
        logger.write(f'{name} data...\n')
        if grouper is None:
            logger.write(f'    n = {len(dataset)}\n')
        else:
            _, group_counts = grouper.metadata_to_group(
                dataset.metadata_array,
                return_counts=True)
            group_counts = group_counts.tolist()
            for group_idx in range(grouper.n_groups):
                logger.write(f'    {grouper.group_str(group_idx)}: n = {group_counts[group_idx]:.0f}\n')
    logger.flush()

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class BatchLogger:
    def __init__(self, csv_path, mode='w', use_wandb=False):
        self.path = csv_path
        self.mode = mode
        self.file = open(csv_path, mode)
        self.is_initialized = False

        # Use Weights and Biases for logging
        self.use_wandb = use_wandb
        if use_wandb:
            self.split = Path(csv_path).stem

    def setup(self, log_dict):
        columns = log_dict.keys()
        # Move epoch and batch to the front if in the log_dict
        for key in ['batch', 'epoch']:
            if key in columns:
                columns = [key] + [k for k in columns if k != key]

        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if self.mode=='w' or (not os.path.exists(self.path)) or os.path.getsize(self.path)==0:
            self.writer.writeheader()
        self.is_initialized = True

    def log(self, log_dict):
        if self.is_initialized is False:
            self.setup(log_dict)
        self.writer.writerow(log_dict)
        self.flush()

        if self.use_wandb:
            results = {}
            for key in log_dict:
                new_key = f'{self.split}/{key}'
                results[new_key] = log_dict[key]
            wandb.log(results)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_config(config, logger):
    for name, val in vars(config).items():
        logger.write(f'{name.replace("_"," ").capitalize()}: {val}\n')
    logger.write('\n')

def initialize_wandb(config):
    name = config.dataset + '_' + config.algorithm + '_' + config.log_dir
    wandb.init(name=name,
               project=f"wilds")
    wandb.config.update(config)

def configure_split_dict(split, data, split_name, train, verbose, grouper, config):
    split_dict = defaultdict(dict)

    # Loaders and dict
    if data is not None: 
        configure_loaders(
            split_dict=split_dict,
            data=data,
            train=train,
            grouper=grouper,
            batch_size=config.batch_size,
            config=config)

    # Set fields
    split_dict['split'] = split
    split_dict['name'] = split_name
    split_dict['verbose'] = verbose 

    # Loggers
    split_dict['eval_logger'] = BatchLogger(
        os.path.join(config.log_dir, f'{split}_eval.csv'), mode=config.mode, use_wandb=(config.use_wandb and verbose))
    split_dict['algo_logger'] = BatchLogger(
        os.path.join(config.log_dir, f'{split}_algo.csv'), mode=config.mode, use_wandb=(config.use_wandb and verbose))
    return split_dict


def configure_loaders(split_dict, data, batch_size, config, train=False, grouper=None):
    """
    modifies split_dict ['dataset'] and ['loader'] keys in place
    Args:
        - data is expected to be a WILDSSubset
        - train is a boolean explaining whether this data will be trained on 
          (e.g. labeled-test is from the test split but trained on)
    """
    split_dict['dataset'] = data
    if train:
        split_dict['loader'] = get_train_loader(
            loader=config.train_loader,
            dataset=split_dict['dataset'],
            batch_size=batch_size,
            uniform_over_groups=config.uniform_over_groups,
            grouper=grouper,
            distinct_groups=config.distinct_groups,
            n_groups_per_batch=config.n_groups_per_batch,
            **config.loader_kwargs)
    else:
        split_dict['loader'] = get_eval_loader(
            loader=config.eval_loader,
            dataset=split_dict['dataset'],
            grouper=grouper,
            batch_size=batch_size,
            **config.loader_kwargs)

def save_pred(y_pred, csv_path):
    df = pd.DataFrame(y_pred.numpy())
    df.to_csv(csv_path, index=False, header=False)

def get_replicate_str(dataset, config):
    if dataset['dataset'].dataset_name == 'poverty':
        replicate_str = f"fold:{config.dataset_kwargs['fold']}"
    else:
        replicate_str = f"seed:{config.seed}"
    return replicate_str

def get_pred_prefix(dataset, config):
    dataset_name = dataset['dataset'].dataset_name
    split = dataset['split']
    replicate_str = get_replicate_str(dataset, config)
    prefix = os.path.join(
        config.log_dir,
        f"{dataset_name}_split:{split}_{replicate_str}_")
    return prefix

def get_model_prefix(dataset, config):
    dataset_name = dataset['dataset'].dataset_name
    replicate_str = get_replicate_str(dataset, config)
    prefix = os.path.join(
        config.log_dir,
        f"{dataset_name}_{replicate_str}_")
    return prefix

def get_indices(reference, values):
    """
    get the first index of each value in values wrt reference, -1 if not found
    assumes all values in values are in reference
    """
    assert set(values).issubset(set(reference))
    sorter = np.argsort(reference)
    return sorter[np.searchsorted(reference, values, sorter=sorter)]
