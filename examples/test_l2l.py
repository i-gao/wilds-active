"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""

import random
import numpy as np
import torch
import learn2learn as l2l
from torch import nn, optim
from types import SimpleNamespace
from collections import defaultdict


import wilds
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.wilds_dataset import WILDSSubset

from utils import set_seed, Logger, log_config, ParseKwargs, load, log_group_data, parse_bool, get_model_prefix, configure_split_dict
from algorithms.initializer import initialize_algorithm
from transforms import initialize_transform
from configs.utils import populate_defaults
from configs.supported import losses
from models.initializer import initialize_model

def sample_task(K, grouper, support_set, labeled_set = None):
    """ 
    Args: 
        - K -- number of labeled shots for adaptation to generate per task
        - grouper
        - support_set -- the WILDSDataset to sample tasks (groups) from
        - labeled_set -- (optional) restrict labeled values to come from this set
    """
    # Sample a task (a single group)
    support_groups = grouper.metadata_to_group(support_set.metadata_array)
    task = np.random.choice(support_groups.unique().numpy())
    print(task)

    if labeled_set is None: labeled_set = support_set
    labeled_groups = grouper.metadata_to_group(labeled_set.metadata_array)

    adaptation_idx = np.random.choice(
        np.arange(len(labeled_set))[labeled_groups == task],
        K, 
        replace=True
    )
    x, y, m = zip(*[labeled_set[i] for i in adaptation_idx])
    adaptation_batch = (torch.stack(x), torch.stack(y), torch.stack(m))

    task_support_set = WILDSSubset(
        support_set.dataset,
        list(set(support_set.indices[support_groups == task]) - set(adaptation_idx)),
        support_set.transform
    )
    evaluation_loader = get_eval_loader(
        loader='standard',
        dataset=task_support_set,
        batch_size=config.batch_size,
        **config.loader_kwargs
    )
    return adaptation_batch, evaluation_loader


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(adaptation_batch, eval_loader, learner, loss, adaptation_steps, shots, ways, device):
    # # Separate data into adaptation/evalutation sets
    # adaptation_indices = np.zeros(data.size(0), dtype=bool)
    # adaptation_indices[np.arange(shots*ways) * 2] = True
    # evaluation_indices = torch.from_numpy(~adaptation_indices)
    # adaptation_indices = torch.from_numpy(adaptation_indices)
    # adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    # evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
 
    # Adapt the model on labeled data
    x, y, _ = adaptation_batch
    for step in range(adaptation_steps):
        train_error = loss.compute(learner(x), y, return_dict=False)
        learner.adapt(train_error)
    
    # Evaluate the adapted model
    epoch_y_true = []
    epoch_y_pred = []
    for x, y, _ in eval_loader: 
        epoch_y_pred.append(learner(x))
        epoch_y_true.append(y)
    epoch_y_true = torch.cat(epoch_y_true)
    epoch_y_pred = torch.cat(epoch_y_pred)
    val_error = loss.compute(epoch_y_pred, epoch_y_true, return_dict=False)
    return val_error


def main(
        config,
        ways=5,
        shots=2,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=3,
        adaptation_steps=1,
        num_iterations=1,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ##############
    # Load datasets

    full_dataset = wilds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)

    train_transform = initialize_transform(
        transform_name=config.train_transform,
        config=config,
        dataset=full_dataset)
    eval_transform = initialize_transform(
        transform_name=config.eval_transform,
        config=config,
        dataset=full_dataset)

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
            datasets[split]['label_manager'] = LabelManager(datasets[split]['dataset'])

    # tasksets = l2l.vision.benchmarks.get_tasksets('omniglot',
    #                                               train_ways=ways,
    #                                               train_samples=2*shots,
    #                                               test_ways=ways,
    #                                               test_samples=2*shots,
    #                                               num_tasks=20000,
    #                                               root='~/data',
    # )

    # train_taskset = WILDSTaskSet(datasets['train']['dataset'], train_grouper, config, meta_train=True)
    # val_taskset = WILDSTaskSet(datasets['val']['dataset'], train_grouper, config, meta_train=True)
    # test_taskset = WILDSTaskSet(datasets['test']['dataset'], train_grouper, config, meta_train=False) 

    ##############
    # Create model
    ## QUESTION: can I just replace model with a wilds model?

    train_dataset = datasets['train']['dataset']
    if (train_dataset.is_classification) and (train_dataset.y_size == 1):
        # For single-task classification, we have one output per class
        d_out = train_dataset.n_classes
    elif (train_dataset.is_classification) and (train_dataset.y_size is None):
        d_out = train_dataset.n_classes
    elif (train_dataset.is_classification) and (train_dataset.y_size > 1) and (train_dataset.n_classes == 2):
        # For multi-task binary classification (each output is the logit for each binary class)
        d_out = train_dataset.y_size
    elif (not train_dataset.is_classification):
        # For regression, we have one output per target dimension
        d_out = train_dataset.y_size

    model = initialize_model(config, d_out)
    # model = l2l.vision.models.OmniglotFC(28 ** 2, ways)
    model.to(device)

    ################

    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False) # this line at least has no bugs
    opt = optim.Adam(maml.parameters(), meta_lr)

    loss = losses[config.loss_function]
    # loss = nn.CrossEntropyLoss(reduction='mean')

    ################

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            adaptation_batch, eval_loader = sample_task(shots, train_grouper, datasets['train']['dataset'])
            # batch = tasksets.train.sample()
            evaluation_error = fast_adapt(adaptation_batch, eval_loader,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            # meta_train_accuracy += evaluation_accuracy.item()

            ##################

            # Compute meta-validation loss
            learner = maml.clone()
            adaptation_batch, eval_loader = sample_task(shots, train_grouper, datasets['val']['dataset'])
            # batch = tasksets.train.sample()
            evaluation_error = fast_adapt(adaptation_batch, eval_loader,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            # meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        # print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        # print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    ###############################

    ## EVAL LOOP
    # finetune for a few steps and then evaluate
    train(
        algorithm=learner,
        datasets=datasets,
        general_logger=logger,
        config=config,
        epoch_offset=0,
        best_val_metric=None,
        train_split="labeled_test",
        val_split=None
    )
    # meta_test_error = 0.0
    # meta_test_accuracy = 0.0
    # for task in range(meta_batch_size):
    #     # Compute meta-testing loss
    #     learner = maml.clone()
    #     batch = test_taskset.sample()
    #     # batch = tasksets.test.sample()
    #     evaluation_error, evaluation_accuracy = fast_adapt(batch,
    #                                                        learner,
    #                                                        loss,
    #                                                        adaptation_steps,
    #                                                        shots,
    #                                                        ways,
    #                                                        device)
    #     meta_test_error += evaluation_error.item()
    #     meta_test_accuracy += evaluation_accuracy.item()
    # print('Meta Test Error', meta_test_error / meta_batch_size)
    # print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == '__main__':
    config = populate_defaults(SimpleNamespace(**{
        'dataset': 'camelyon17',
        'version': None,
        'frac': 0.0001,
        'download': False,
        'algorithm': 'ERM',
        'root_dir': '/u/scr/nlp/wilds/data',
        'dataset_kwargs': {},
        'loader_kwargs': {},
        'model_kwargs': {},
        'optimizer_kwargs': {},
        'scheduler_kwargs': {},
        'no_group_logging': True,
        'distinct_groups': True,
        'log_dir': '../../test_run',
        'mode': 'w',
        'use_wandb': False,
        'active_learning': False,
    }))
    main(config)
