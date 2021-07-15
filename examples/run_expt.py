import os, csv
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
from collections import defaultdict

# TODO: This is needed to test the WILDS package locally. Remove later -Tony
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from utils import set_seed, Logger, log_config, ParseKwargs, load, log_group_data, parse_bool, get_model_prefix, configure_split_dict
from train import train, evaluate
from algorithms.initializer import initialize_algorithm
from active import run_active_learning, LabelManager
from selection_fn import initialize_selection_function
from few_shot import initialize_few_shot_algorithm
from transforms import initialize_transform
from configs.utils import populate_defaults
import configs.supported as supported

def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
    parser.add_argument('--pretrained_model_path', default=None, type=str, help="Specify a path to a pretrained model's weights")

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to download the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str)

    # Unlabeled Dataset
    parser.add_argument('--unlabeled_split', default=None, type=str, help='Unlabeled split to use')
    parser.add_argument('--unlabeled_version', default=None, type=str)

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--unlabeled_n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--unlabeled_batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')

    # Active Learning
    parser.add_argument('--active_learning', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--selection_function', choices=supported.selection_functions)
    parser.add_argument('--selection_function_kwargs', nargs='*', action=ParseKwargs, default={}, help="keyword arguments for selection fn passed as key1=value1 key2=value2")
    parser.add_argument('--n_rounds', type=int, default=1, help="number of times to repeat the selection-train cycle")
    parser.add_argument('--selectby_fields', nargs='+', help="If set, acts like a grouper and n_shots are acquired per selection group (e.g. y x hospital selects K examples per y x hospital).")
    parser.add_argument('--n_shots', type=int, help="number of shots (labels) to actively acquire each round")
    parser.add_argument('--few_shot_algorithm', choices=supported.few_shot_algorithms)
    parser.add_argument('--few_shot_kwargs', nargs='*', action=ParseKwargs, default={},
        help='keyword arguments for few shot algorithm initialization passed as key1=value1 key2=value2')

    # Model
    parser.add_argument('--model', choices=supported.models)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
        help='keyword arguments for model initialization passed as key1=value1 key2=value2')

    # Transforms
    parser.add_argument('--train_transform', choices=supported.transforms)
    parser.add_argument('--eval_transform', choices=supported.transforms)
    parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--resize_scale', type=float)
    parser.add_argument('--max_token_length', type=int)
    parser.add_argument('--randaugment_n', type=int, help='N parameter of RandAugment - the number of transformations to apply.')
    parser.add_argument('--randaugment_m', type=int,
        help='M parameter of RandAugment - the magnitude of the transformation. Values range from 1 to 10, where 10 indicates the maximum scale for a transformation.')

    # Objective
    parser.add_argument('--loss_function', choices = supported.losses)

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--maml_first_order', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--metalearning_k', type=int)
    parser.add_argument('--metalearning_adapt_lr', type=float)
    parser.add_argument('--metalearning_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--self_training_lambda', type=float)
    parser.add_argument('--self_training_threshold', type=float)
    parser.add_argument('--algo_log_metric')

    # Model selection
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={})

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')
    parser.add_argument('--eval_additional_every', default=1, type=int, help='Eval additional splits every _ epochs.')

    # Misc
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

    config = parser.parse_args()
    config = populate_defaults(config)

    # set device
    config.device = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu")

    ## Initialize logs
    if os.path.exists(config.log_dir) and config.resume:
        resume=True
        config.mode='a'
    elif os.path.exists(config.log_dir) and config.eval_only:
        resume=False
        config.mode='a'
    else:
        resume=False
        config.mode='w'

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    logger = Logger(os.path.join(config.log_dir, 'log.txt'), config.mode)

    # Record config
    log_config(config, logger)

    # Set random seed
    set_seed(config.seed)

    # Data
    full_dataset = wilds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)

    # To implement data augmentation (i.e., have different transforms
    # at training time vs. test time), modify these two lines:
    train_transform = initialize_transform(
        transform_name=config.train_transform,
        config=config,
        dataset=full_dataset)
    eval_transform = initialize_transform(
        transform_name=config.eval_transform,
        config=config,
        dataset=full_dataset)

    if config.algorithm == "fixmatch":
        # For FixMatch, we need our loader to return batches in the form ((x_weak, x_strong), m)
        # We do this by initializing a special transform function
        unlabeled_train_transform = initialize_transform(
            config.train_transform, config, full_dataset, additional_transform_name="fixmatch"
        )

    train_grouper = CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=config.groupby_fields)

    datasets = defaultdict(dict)
    for split in full_dataset.split_dict.keys():
        if split=='train':
            transform = train_transform
            verbose = True
        elif split == 'val':
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = False

        data = full_dataset.get_subset(
            split,
            frac=config.frac,
            transform=transform)
        
        datasets[split] = configure_split_dict(
            data=data,
            split=split,
            split_name=full_dataset.split_names[split],
            train=(split=='train'),
            verbose=verbose,
            grouper=train_grouper,
            config=config)
 
        if 'test' in split and config.active_learning:
            # Perform active learning on the standard test split
            datasets[split]['label_manager'] = LabelManager(
                datasets[split]['dataset'],
                train_transform,
                eval_transform,
                unlabeled_train_transform=unlabeled_train_transform
            )

    if config.use_wandb:
        initialize_wandb(config)

    # Logging dataset info
    # Show class breakdown if feasible
    if config.no_group_logging and full_dataset.is_classification and full_dataset.y_size==1 and full_dataset.n_classes <= 10:
        log_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['y'])
    elif config.no_group_logging:
        log_grouper = None
    else:
        log_grouper = train_grouper
    log_group_data(datasets, log_grouper, logger)

    ## Initialize algorithm
    algorithm = initialize_algorithm(
        config=config,
        datasets=datasets,
        train_grouper=train_grouper)

    # Load pretrained weights if specified (this can be overriden by resume)
    if config.pretrained_model_path is not None and os.path.exists(config.pretrained_model_path):
        # The full model name is expected to be specified, so just load.
        try:
            prev_epoch, best_val_metric = load(algorithm, config.pretrained_model_path, device=config.device)
            epoch_offset = 0
            logger.write(
                (f'Initialized algorithm with pretrained weights from {config.pretrained_model_path} ')
                + (f'previously trained for {prev_epoch} epochs ' if prev_epoch else '')
                + (f'with previous val metric {best_val_metric} ' if best_val_metric else '')
            )
        except:
            pass

    # Resume from most recent model in log_dir
    model_prefix = get_model_prefix(datasets['train'], config)
    if not config.eval_only:
        ## If doing active learning, expects to load a model trained on source
        resume_success = False
        if config.resume:
            save_path = model_prefix + 'epoch:last_model.pth'
            if not os.path.exists(save_path):
                epochs = [
                    int(file.split('epoch:')[1].split('_')[0])
                    for file in os.listdir(config.log_dir) if file.endswith('.pth')]
                if len(epochs) > 0:
                    latest_epoch = max(epochs)
                    save_path = model_prefix + f'epoch:{latest_epoch}_model.pth'
            try:
                prev_epoch, best_val_metric = load(algorithm, save_path, config.device)
                epoch_offset = prev_epoch + 1
                logger.write(f'Resuming from epoch {epoch_offset} with best val metric {best_val_metric}')
                resume_success = True
            except FileNotFoundError:
                pass

        if resume_success == False:
            epoch_offset=0
            best_val_metric=None

        if config.active_learning:
            few_shot_algorithm = initialize_few_shot_algorithm(config, algorithm)
            select_grouper = CombinatorialGrouper(
                dataset=full_dataset,
                groupby_fields=config.selectby_fields)
            selection_fn = initialize_selection_function(config, algorithm, few_shot_algorithm, select_grouper, algo_grouper=train_grouper)
            run_active_learning(
                selection_fn=selection_fn,
                few_shot_algorithm=few_shot_algorithm,
                datasets=datasets,
                general_logger=logger,
                grouper=train_grouper,
                config=config,
                full_dataset=full_dataset)
        else: 
            train(
                algorithm=algorithm,
                datasets=datasets,
                general_logger=logger,
                config=config,
                epoch_offset=epoch_offset,
                best_val_metric=best_val_metric)
    else:
        if config.eval_epoch is None:
            eval_model_path = model_prefix + 'epoch:best_model.pth'
        else:
            eval_model_path = model_prefix +  f'epoch:{config.eval_epoch}_model.pth'
        best_epoch, best_val_metric = load(algorithm, eval_model_path, config.device)
        if config.eval_epoch is None:
            epoch = best_epoch
        else:
            epoch = config.eval_epoch
        evaluate(
            algorithm=algorithm,
            datasets=datasets,
            epoch=epoch,
            general_logger=logger,
            config=config)

    logger.close()
    for split in datasets:
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()

if __name__=='__main__':
    main()
