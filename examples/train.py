import os
from tqdm import tqdm
import torch
from utils import save_model, save_pred, get_pred_prefix, get_model_prefix
import torch.autograd.profiler as profiler
from configs.supported import process_outputs_functions
import numpy as np

from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.wilds_dataset import WILDSSubset

def run_maml_epoch(algorithm, dataset, general_logger, epoch, config, train=False, labeled_set=None):
    if general_logger and dataset['verbose']:
        general_logger.write(f"\n{dataset['name']}:\n")

    # meta-training on tasks
    if train: 
        algorithm.train()
        for _ in range(config.maml_meta_batch_size):
            task, adaptation_batch = sample_maml_task(
                config.maml_k,
                algorithm.grouper,
                dataset['dataset'],
                config.batch_size,
                config.loader_kwargs,
                labeled_set=labeled_set
            )
            general_logger.write(f'Sampled task {task}\n')
            algorithm.adapt_task(adaptation_batch, dataset['loader'])
    
    # finetune and then evaluate
    adapt_data = labeled_set['loader'] if labeled_set else dataset['loader']
    # need to convert adapt_data -> tensor
    _, adaptation_batch = sample_maml_task(
        config.maml_k,
        None,
        dataset['dataset'],
        config.batch_size,
        config.loader_kwargs,
        labeled_set=labeled_set
    )

    epoch_results = algorithm.evaluate(adaptation_batch, dataset['loader']) 
    
    epoch_y_pred = epoch_results['y_pred'].clone().detach()
    if config.process_outputs_function is not None:
        epoch_y_pred = process_outputs_functions[config.process_outputs_function](epoch_y_pred)
    
    results, results_str = dataset['dataset'].eval(
        epoch_y_pred,
        epoch_results['y_true'],
        epoch_results['metadata']
    )

    if config.scheduler_metric_split==dataset['split']:
        algorithm.step_schedulers(
            is_epoch=True,
            metrics=results,
            log_access=False
        )

    # log after updating the scheduler in case it needs to access the internal logs
    if general_logger: log_results(algorithm, dataset, general_logger, epoch, 0)
    results['epoch'] = epoch
    dataset['eval_logger'].log(results)
    if general_logger and dataset['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)
    return results, epoch_y_pred
    
def run_epoch(algorithm, dataset, general_logger, epoch, config, train):
    if general_logger and dataset['verbose']:
        general_logger.write(f"\n{dataset['name']}:\n")

    if train:
        algorithm.train()
    else:
        algorithm.eval()

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment batch_idx
    batch_idx = 0
    iterator = tqdm(dataset['loader']) if config.progress_bar else dataset['loader']

    for batch in iterator:
        if train:
            batch_results = algorithm.update(batch)
        else:
            batch_results = algorithm.evaluate(batch)

        # These tensors are already detached, but we need to clone them again
        # Otherwise they don't get garbage collected properly in some versions
        # The subsequent detach is just for safety
        # (they should already be detached in batch_results)
        epoch_y_true.append(batch_results['y_true'].clone().detach())
        y_pred = batch_results['y_pred'].clone().detach()
        if config.process_outputs_function is not None:
            y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
        epoch_y_pred.append(y_pred)
        epoch_metadata.append(batch_results['metadata'].clone().detach())

        if general_logger and train and (batch_idx+1) % config.log_every==0:
            log_results(algorithm, dataset, general_logger, epoch, batch_idx)

        batch_idx += 1

    epoch_y_pred = torch.cat(epoch_y_pred)
    epoch_y_true = torch.cat(epoch_y_true)
    epoch_metadata = torch.cat(epoch_metadata)
    results, results_str = dataset['dataset'].eval(
        epoch_y_pred,
        epoch_y_true,
        epoch_metadata)

    if config.scheduler_metric_split==dataset['split']:
        algorithm.step_schedulers(
            is_epoch=True,
            metrics=results,
            log_access=(not train))

    # log after updating the scheduler in case it needs to access the internal logs
    if general_logger: log_results(algorithm, dataset, general_logger, epoch, batch_idx)

    results['epoch'] = epoch
    dataset['eval_logger'].log(results)
    if general_logger and dataset['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)

    return results, epoch_y_pred


def train(algorithm, datasets, general_logger, config, epoch_offset, best_val_metric, train_split="train", val_split="val", rnd=None):
    for epoch in range(epoch_offset, config.n_epochs):
        general_logger.write('\nEpoch [%d]:\n' % epoch)

        epoch_fn = run_maml_epoch if config.algorithm == 'MAML' else run_epoch

        # First run training
        epoch_fn(algorithm, datasets[train_split], general_logger, epoch, config, train=True)

        # Then run val
        if val_split is None: 
            is_best = False # only save last
            best_val_metric = None
        else:
            val_results, y_pred = epoch_fn(algorithm, datasets[val_split], general_logger, epoch, config, train=False)
            curr_val_metric = val_results[config.val_metric]
            general_logger.write(f'Validation {config.val_metric}: {curr_val_metric:.3f}\n')

            if best_val_metric is None:
                is_best = True
            else:
                if config.val_metric_decreasing:
                    is_best = curr_val_metric < best_val_metric
                else:
                    is_best = curr_val_metric > best_val_metric
            if is_best:
                best_val_metric = curr_val_metric
                general_logger.write(f'Epoch {epoch} has the best validation performance so far.\n')
            save_pred_if_needed(y_pred, datasets[train_split], epoch, rnd, config, is_best)
        save_model_if_needed(algorithm, datasets[train_split], epoch, rnd, config, is_best, best_val_metric)
       
        # Then run everything else
        if config.evaluate_all_splits:
            additional_splits = [split for split in datasets.keys() if split not in ['train','val']]
        else:
            additional_splits = config.eval_splits
        if epoch % config.eval_additional_every == 0 or epoch+1 == config.n_epochs:
            for split in additional_splits:
                _, y_pred = epoch_fn(algorithm, datasets[split], general_logger, epoch, config, train=False)
                save_pred_if_needed(y_pred, datasets[split], epoch, rnd, config, is_best)

        general_logger.write('\n')


def evaluate(algorithm, datasets, epoch, general_logger, config):
    algorithm.eval()
    for split, dataset in datasets.items():
        if (not config.evaluate_all_splits) and (split not in config.eval_splits):
            continue
        epoch_y_true = []
        epoch_y_pred = []
        epoch_metadata = []
        iterator = tqdm(dataset['loader']) if config.progress_bar else dataset['loader']
        for batch in iterator:
            batch_results = algorithm.evaluate(batch)
            epoch_y_true.append(batch_results['y_true'].clone().detach())
            y_pred = batch_results['y_pred'].clone().detach()
            if config.process_outputs_function is not None:
                y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
            epoch_y_pred.append(y_pred)
            epoch_metadata.append(batch_results['metadata'].clone().detach())

        results, results_str = dataset['dataset'].eval(
            torch.cat(epoch_y_pred),
            torch.cat(epoch_y_true),
            torch.cat(epoch_metadata))

        results['epoch'] = epoch
        dataset['eval_logger'].log(results)
        general_logger.write(f'Eval split {split} at epoch {epoch}:\n')
        general_logger.write(results_str)

        # Skip saving train preds, since the train loader generally shuffles the data
        if split != 'train':
            save_pred_if_needed(y_pred, dataset, epoch, config, is_best=False, force_save=True)

def sample_maml_task(K, grouper, support_set, batch_size, loader_kwargs, labeled_set=None, enforce_disjoint=False):
    """ 
    Args: 
        - K -- number of labeled shots for adaptation to generate per task
        - grouper
        - support_set -- the WILDSDataset to sample tasks (groups) from
        - labeled_set -- (optional) restrict labeled values to come from this set
    """
    if labeled_set is None: labeled_set = support_set
    if grouper is None:
        # Sample k random points
        adaptation_idx = np.random.choice(
            np.arange(len(labeled_set)),
            K, 
            replace=True
        )
        task=None      
    else:
        # Sample a task (a single group)
        support_groups = grouper.metadata_to_group(support_set.metadata_array)
        task = np.random.choice(support_groups.unique().numpy()) 
        labeled_groups = grouper.metadata_to_group(labeled_set.metadata_array)

        adaptation_idx = np.random.choice(
            np.arange(len(labeled_set))[labeled_groups == task],
            K, 
            replace=True
        )

    x, y, m = zip(*[labeled_set[i] for i in adaptation_idx])
    adaptation_batch = (torch.stack(x), torch.stack(y), torch.stack(m))

    if enforce_disjoint:
        task_support_set = WILDSSubset(
            support_set.dataset,
            list(set(support_set.indices[support_groups == task]) - set(adaptation_idx)),
            support_set.transform
        )
        evaluation_loader = get_eval_loader(
            loader='standard',
            dataset=task_support_set,
            batch_size=batch_size,
            **loader_kwargs
        )
        return task, adaptation_batch, evaluation_loader
    else:
        return task, adaptation_batch

def log_results(algorithm, dataset, general_logger, epoch, batch_idx):
    if algorithm.has_log:
        log = algorithm.get_log()
        log['epoch'] = epoch
        log['batch'] = batch_idx
        dataset['algo_logger'].log(log)
        if dataset['verbose']:
            general_logger.write(algorithm.get_pretty_log_str())
        algorithm.reset_log()


def save_pred_if_needed(y_pred, dataset, epoch, rnd, config, is_best, force_save=False):
    round_str = f'round:{rnd}_' if config.active_learning else ''
    if config.save_pred:
        prefix = get_pred_prefix(dataset, config)
        if force_save or (config.save_step is not None and (epoch + 1) % config.save_step == 0):
            save_pred(y_pred, prefix + f'{round_str}epoch:{epoch}_pred.csv')
        if config.save_last:
            save_pred(y_pred, prefix + f'{round_str}epoch:last_pred.csv')
        if config.save_best and is_best:
            save_pred(y_pred, prefix + f'{round_str}epoch:best_pred.csv')


def save_model_if_needed(algorithm, dataset, epoch, rnd, config, is_best, best_val_metric):
    round_str = f'round:{rnd}_' if config.active_learning else ''
    prefix = get_model_prefix(dataset, config)
    if config.save_step is not None and (epoch + 1) % config.save_step == 0:
        save_model(algorithm, epoch, best_val_metric, prefix + f'{round_str}epoch:{epoch}_model.pth')
    if config.save_last:
        save_model(algorithm, epoch, best_val_metric, prefix + f'{round_str}epoch:last_model.pth')
    if config.save_best and is_best:
        save_model(algorithm, epoch, best_val_metric, prefix + f'{round_str}epoch:best_model.pth')
