import torch
import torch.nn as nn
import os

from models.layers import Identity
from utils import load

def initialize_model(config, d_out, is_featurizer=False):
    """
    Initializes models according to the config
        Args:
            - config (dictionary): config dictionary
            - d_out (int): the dimensionality of the model output
            - is_featurizer (bool): whether to return a model or a (featurizer, classifier) pair that constitutes a model.
        Output:
            If is_featurizer=True:
            - featurizer: a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality)
            - classifier: a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear layer.

            If is_featurizer=False:
            - model: a model that is equivalent to nn.Sequential(featurizer, classifier)

        Pretrained weights are loaded according to config.pretrained_model_path using either transformers.from_pretrained (for bert-based models)
        or our own utils.load function (for torchvision models, resnet18-ms, and gin-virtual). 
        There is currently no support for loading pretrained weights from disk for other models.
    """
    if config.model in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'wideresnet50', 'densenet121'):
        if is_featurizer:
            featurizer = initialize_torchvision_model(
                name=config.model,
                d_out=None,
                **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_torchvision_model(
                name=config.model,
                d_out=d_out,
                **config.model_kwargs)

    elif 'bert' in config.model:
        if is_featurizer:
            featurizer = initialize_bert_based_model(config, d_out, is_featurizer)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_bert_based_model(config, d_out)

    elif config.model == 'resnet18_ms':  # multispectral resnet 18
        from models.resnet_multispectral import ResNet18
        if is_featurizer:
            featurizer = ResNet18(num_classes=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = ResNet18(num_classes=d_out, **config.model_kwargs)

    elif config.model == 'gin-virtual':
        from models.gnn import GINVirtual
        if is_featurizer:
            featurizer = GINVirtual(num_tasks=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = GINVirtual(num_tasks=d_out, **config.model_kwargs)

    elif config.model == 'code-gpt-py':
        from models.code_gpt import GPT2LMHeadLogit, GPT2FeaturizerLMHeadLogit
        from transformers import GPT2Tokenizer
        name = 'microsoft/CodeGPT-small-py'
        tokenizer = GPT2Tokenizer.from_pretrained(name)
        if is_featurizer:
            model = GPT2FeaturizerLMHeadLogit.from_pretrained(name)
            model.resize_token_embeddings(len(tokenizer))
            featurizer = model.transformer
            classifier = model.lm_head
            model = (featurizer, classifier)
        else:
            model = GPT2LMHeadLogit.from_pretrained(name)
            model.resize_token_embeddings(len(tokenizer))

    elif config.model == 'logistic_regression':
        assert not is_featurizer, "Featurizer not supported for logistic regression"
        model = nn.Linear(out_features=d_out, **config.model_kwargs)

    else:
        raise ValueError(f'Model: {config.model} not recognized.')

    # Load pretrained weights from disk using our utils.load function
    # This has only been tested on some models (mostly vision), so run this code iff we're sure it works
    # We've already loaded pretrained weights for bert-based models using the transformers library 
    if config.model not in ('code-gpt-py', 'logistic_regression', 'unet-seq', 'fasterrcnn') and 'bert' not in config.model:
        if (config.pretrained_model_path and os.path.exists(config.pretrained_model_path): 
            try:
                if type(model) is tuple: 
                    # load both featurizer and classifier
                    prev_epoch, best_val_metric = load(
                        nn.Sequential(*model), 
                        config.pretrained_model_path, device=config.device
                    )
                else: 
                    prev_epoch, best_val_metric = load(model, config.pretrained_model_path, device=config.device)

                print(
                    (f'Initialized model with pretrained weights from {config.pretrained_model_path} ')
                    + (f'previously trained for {prev_epoch} epochs ' if prev_epoch else '')
                    + (f'with previous val metric {best_val_metric} ' if best_val_metric else '')
                )
            except:
                print('Something went wrong loading the pretrained model.')
                pass

    return model


def initialize_bert_based_model(config, d_out, is_featurizer=False):
    from models.bert.bert import BertClassifier, BertFeaturizer
    from models.bert.distilbert import DistilBertClassifier, DistilBertFeaturizer

    if config.pretrained_model_path:
        print(f'Initialized model with pretrained weights from {config.pretrained_model_path}')
        config.model_kwargs['state_dict'] = torch.load(config.pretrained_model_path, map_location=config.device)

    if config.model == 'bert-base-uncased':
        if is_featurizer:
            model = BertFeaturizer.from_pretrained(config.model, **config.model_kwargs)
        else:
            model = BertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)
    elif config.model == 'distilbert-base-uncased':
        if is_featurizer:
            model = DistilBertFeaturizer.from_pretrained(config.model, **config.model_kwargs)
        else:
            model = DistilBertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)
    else:
        raise ValueError(f'Model: {config.model} not recognized.')
    return model

def initialize_torchvision_model(name, d_out, **kwargs):
    import torchvision

    # get constructor and last layer names
    if name == 'wideresnet50':
        constructor_name = 'wide_resnet50_2'
        last_layer_name = 'fc'
    elif name == 'densenet121':
        constructor_name = name
        last_layer_name = 'classifier'
    elif name in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
        constructor_name = name
        last_layer_name = 'fc'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else: # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)

    return model