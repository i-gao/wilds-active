import os
from tqdm import tqdm
import math
import torch
from utils import save_model, save_array, get_pred_prefix, get_model_prefix, InfiniteDataIterator
import torch.autograd.profiler as profiler
from configs.supported import process_outputs_functions
from algorithms.metalearning import sample_metalearning_task

from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.wilds_dataset import WILDSSubset

def run_metalearning_epoch(algorithm, dataset, general_logger, epoch, config, train=False, labeled_set=None, unlabeled_dataset=None):
    if dataset['verbose']:
        general_logger.write(f"\n{dataset['name']}:\n")

    # meta-training on tasks
    if train: 
        algorithm.train()
        for _ in range(config.metalearning_kwargs.get('meta_batch_size')):
            task, adaptation_batch, evaluation_batch = sample_metalearning_task(
                K=config.metalearning_k,
                M=config.metalearning_kwargs.get('n_eval_examples'),
                grouper=algorithm.grouper,
                n_groups_task=config.metalearning_kwargs.get('n_groups_task'),
                support_set=dataset['dataset'],
                labeled_set=labeled_set
            )
            general_logger.write(f'Sampled task {task}\n')
            algorithm.adapt_task(adaptation_batch, evaluation_batch)
    
    # finetune and then evaluate
    adapt_data = labeled_set['train_loader'] if labeled_set else dataset['train_loader']
    # need to convert adapt_data -> tensor
    _, adaptation_batch, _ = sample_metalearning_task(
        K=config.metalearning_k,
        M=config.metalearning_kwargs.get('n_eval_examples'),
        grouper=None,
        n_groups_task=None,
        support_set=dataset['dataset'],
        labeled_set=labeled_set
    )
    _, adaptation_batch_groups = algorithm.grouper.metadata_to_group(adaptation_batch[2]).unique(return_counts=True)# TODO: remove
    general_logger.write(f'Sampled for evaluation batch with groups {adaptation_batch_groups}\n') # TODO: remove

    epoch_results = algorithm.evaluate(adaptation_batch, dataset['eval_loader']) 
    
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
    log_results(algorithm, dataset, general_logger, epoch, 0)
    results['epoch'] = epoch
    dataset['eval_logger'].log(results)
    if dataset['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)
    return results, epoch_y_pred, None
    
def run_epoch(algorithm, dataset, general_logger, epoch, config, train, unlabeled_dataset=None):
    if dataset['verbose']:
        general_logger.write(f"\n{dataset['name']}:\n")

    if train:
        algorithm.train()
        loader_name = 'train_loader'
    else:
        algorithm.eval()
        loader_name = 'eval_loader'

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = []
    epoch_y_pseudo = []
    epoch_metadata = []

    # Assert that data loaders are defined for the datasets
    assert loader_name in dataset, "A data loader must be defined for the dataset."
    if unlabeled_dataset:
        assert loader_name in unlabeled_dataset, "A data loader must be defined for the dataset."

    batches = dataset[loader_name]
    if config.progress_bar:
        batches = tqdm(batches)
    last_batch_idx = len(batches)-1

    if unlabeled_dataset: unlabeled_batches = InfiniteDataIterator(unlabeled_dataset[loader_name])

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment batch_idx
    batch_idx = 0
    for labeled_batch in batches:
        algo_fn = algorithm.update if train else algorithm.evaluate
        if unlabeled_dataset:
            unlabeled_batch = next(unlabeled_batches)
            batch_results = algo_fn(labeled_batch, unlabeled_batch, is_epoch_end=(batch_idx==last_batch_idx))
        else:
            batch_results = algo_fn(labeled_batch, is_epoch_end=(batch_idx==last_batch_idx))

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
        if 'unlabeled_y_pseudo' in batch_results: epoch_y_pseudo.append(batch_results['unlabeled_y_pseudo'].clone().detach())
        
        if train: 
            effective_batch_idx = (batch_idx + 1) / config.gradient_accumulation_steps
        else: 
            effective_batch_idx = batch_idx + 1

        if train and effective_batch_idx % config.log_every==0:
            log_results(algorithm, dataset, general_logger, epoch, math.ceil(effective_batch_idx))

        batch_idx += 1

    epoch_y_pred = torch.cat(epoch_y_pred)
    epoch_y_true = torch.cat(epoch_y_true)
    epoch_y_pseudo = torch.cat(epoch_y_pseudo) if len(epoch_y_pseudo) > 0 else None
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
    log_results(algorithm, dataset, general_logger, epoch, math.ceil(effective_batch_idx))

    results['epoch'] = epoch
    dataset['eval_logger'].log(results)
    if dataset['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)

    return results, epoch_y_pred, epoch_y_pseudo


def train(algorithm, datasets, general_logger, config, epoch_offset, best_val_metric, train_split="train", val_split="val", unlabeled_split=None):
    for epoch in range(epoch_offset, config.n_epochs):
        general_logger.write('\nEpoch [%d]:\n' % epoch)

        epoch_fn = run_metalearning_epoch if config.algorithm in ['MAML', 'ANIL'] else run_epoch

        # First run training
        unlabeled_dataset = datasets[unlabeled_split] if unlabeled_split else None
        epoch_fn(algorithm, datasets[train_split], general_logger, epoch, config, train=True, unlabeled_dataset=unlabeled_dataset)

        # Then run val
        if val_split is None: 
            is_best = False # only save last
            best_val_metric = None
        else:
            val_results, y_pred, _ = epoch_fn(algorithm, datasets[val_split], general_logger, epoch, config, train=False)
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
            save_pred_if_needed(y_pred, val_split, datasets[val_split], epoch, config, is_best)
        save_model_if_needed(algorithm, datasets[train_split], epoch, config, is_best, best_val_metric)
       
        # Then run everything else
        if config.evaluate_all_splits:
            additional_splits = [
                split for split in datasets.keys() if split not in ['train','val',f'labeled_{config.target_split}',f'unlabeled_{config.target_split}_augmented']
            ]
        else:
            additional_splits = config.eval_splits
        if epoch % config.eval_additional_every == 0 or epoch+1 == config.n_epochs:
            for split in additional_splits:
                if split == f'unlabeled_{config.target_split}':
                    _, y_pred, y_pseudo = epoch_fn(algorithm, datasets[split], general_logger, epoch, config, train=False, unlabeled_dataset=datasets[f'unlabeled_{config.target_split}_augmented'])
                else:
                    _, y_pred, y_pseudo = epoch_fn(algorithm, datasets[split], general_logger, epoch, config, train=False)
                save_pred_if_needed(y_pred, split, datasets[split], epoch, config, is_best)
                save_pseudo_if_needed(y_pseudo, split, datasets[split], epoch, config, is_best)
        general_logger.write('\n')


def evaluate(algorithm, datasets, epoch, general_logger, config):
    algorithm.eval()
    for split, dataset in datasets.items():
        if (not config.evaluate_all_splits) and (split not in config.eval_splits):
            continue
        epoch_y_true = []
        epoch_y_pred = []
        epoch_metadata = []
        iterator = tqdm(dataset['eval_loader']) if config.progress_bar else dataset['eval_loader']
        for batch in iterator:
            batch_results = algorithm.evaluate(batch)
            epoch_y_true.append(batch_results['y_true'].clone().detach())
            y_pred = batch_results['y_pred'].clone().detach()
            if config.process_outputs_function is not None:
                y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
            epoch_y_pred.append(y_pred)
            epoch_metadata.append(batch_results['metadata'].clone().detach())

        epoch_y_pred = torch.cat(epoch_y_pred)
        epoch_y_true = torch.cat(epoch_y_true)
        epoch_metadata = torch.cat(epoch_metadata)
        results, results_str = dataset['dataset'].eval(
            epoch_y_pred,
            epoch_y_true,
            epoch_metadata)

        results['epoch'] = epoch
        dataset['eval_logger'].log(results)
        general_logger.write(f'Eval split {split} at epoch {epoch}:\n')
        general_logger.write(results_str)

        # Skip saving train preds, since the train loader generally shuffles the data
        if split != 'train':
            save_pred_if_needed(epoch_y_pred, split, dataset, epoch, config, is_best=(config.eval_epoch is None), force_save=True)

def infer_predictions(model, loader, config):
    """
    Simple inference loop that performs inference using a model (not algorithm) and returns model outputs.
    Compatible with both labeled and unlabeled WILDS datasets.
    """
    model.eval()
    y_pred = []
    iterator = tqdm(loader) if config.progress_bar else loader
    for batch in iterator:
        x = batch[0]
        x = x.to(config.device)
        with torch.no_grad(): 
            output = model(x)
            if not config.soft_pseudolabels and config.process_outputs_function is not None:
                output = process_outputs_functions[config.process_outputs_function](output)
            elif config.soft_pseudolabels:
                output = torch.nn.functional.softmax(output, dim=1)
        y_pred.append(output.clone().detach())
    return torch.cat(y_pred, 0).to(torch.device('cpu'))

def log_results(algorithm, dataset, general_logger, epoch, effective_batch_idx):
    if algorithm.has_log:
        log = algorithm.get_log()
        log['epoch'] = epoch
        log['batch'] = effective_batch_idx
        dataset['algo_logger'].log(log)
        if dataset['verbose']:
            general_logger.write(algorithm.get_pretty_log_str())
        algorithm.reset_log()

def save_pred_if_needed(y_pred, split, dataset, epoch, config, is_best, force_save=False):
    if (not config.save_pred_step) or (split not in config.save_splits):
        return
    prefix = get_pred_prefix(dataset, config)
    if force_save or (config.save_pred_step is not None and (epoch + 1) % config.save_pred_step == 0):
        save_array(y_pred, prefix + f'epoch:{epoch}_pred.csv')
    if config.save_last:
        save_array(y_pred, prefix + f'epoch:last_pred.csv')
    if config.save_best and is_best:
        save_array(y_pred, prefix + f'epoch:best_pred.csv')

def save_pseudo_if_needed(y_pseudo, split, dataset, epoch, config, is_best, force_save=False):
    if (not config.save_pseudo_step) or (y_pseudo is None) or (split not in config.save_splits):
        return
    prefix = get_pred_prefix(dataset, config)
    if config.algorithm == 'NoisyStudent': # save on first epoch; pseudolabels are constant
        save_array(y_pseudo, prefix + f'pseudo.csv')
    else: 
        if force_save or (config.save_pseudo_step is not None and (epoch + 1) % config.save_pseudo_step == 0):
            save_array(y_pseudo, prefix + f'epoch:{epoch}_pseudo.csv')
        if config.save_last:
            save_array(y_pseudo, prefix + f'epoch:last_pseudo.csv')
        if config.save_best and is_best:
            save_array(y_pseudo, prefix + f'epoch:best_pseudo.csv')


def save_model_if_needed(algorithm, dataset, epoch, config, is_best, best_val_metric):
    prefix = get_model_prefix(dataset, config)
    if config.save_model_step is not None and (epoch + 1) % config.save_model_step == 0:
        save_model(algorithm, epoch, best_val_metric, prefix + f'epoch:{epoch}_model.pth')
    if config.save_last:
        save_model(algorithm, epoch, best_val_metric, prefix + f'epoch:last_model.pth')
    if config.save_best and is_best:
        save_model(algorithm, epoch, best_val_metric, prefix + f'epoch:best_model.pth')
