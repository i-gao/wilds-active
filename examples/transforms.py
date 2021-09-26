import copy
from typing import List

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import BertTokenizerFast, DistilBertTokenizerFast

from data_augmentation.randaugment import FIX_MATCH_AUGMENTATION_POOL, RandAugment


_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

def initialize_transform(
    transform_name, config, dataset, is_training, additional_transform=None
):
    """
    By default, transforms should take in `x` and return `transformed_x`.
    For transforms that take in `(x, y)` and return `(transformed_x, transformed_y)`,
    set `do_transform_y` to True when initializing the WILDSSubset.    
    """
    if transform_name is None:
        return None
    elif transform_name == "bert":
        return initialize_bert_transform(config)

    # For images
    should_rgb_transform = False
    if transform_name == "image_base":
        normalize = True
        base_transform_steps = get_image_base_transform_steps(config, dataset)
    elif transform_name == "image_resize":
        normalize = True
        base_transform_steps = get_image_resize_transform_steps(
            config, dataset
        )
    elif transform_name == "image_resize_and_center_crop":
        normalize = True
        base_transform_steps = get_image_resize_and_center_crop_transform_steps(
            config, dataset
        )
    elif transform_name == "poverty":
        if not is_training:
            return None
        normalize = False
        should_rgb_transform = True
        base_transform_steps = get_poverty_train_transform_steps()
    else:
        raise ValueError(f"{transform_name} not recognized")

    default_normalization = transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
    )

    ## Additional transforms    
    ## Optionally layer on stochastic transforms
    additional_transform_functions = {
        'randaugment': add_rand_augment_transform,
        'weak': add_weak_transform,
        None: add_no_transform, # adds no additional transform; just stacks the base transform steps + normalization
    } 
    if type(additional_transform) == list and len(additional_transform) == 1: 
        additional_transform = additional_transform[0]
    
    if type(additional_transform) == list: 
        # Has the loader return tuples of differently augmented views of x, e.g. for fixmatch
        transformations = tuple(
            additional_transform_functions[name](config, dataset, base_transform_steps, normalize, default_normalization) for name in additional_transform
        )
        transform = MultipleTransforms(transformations)
    else:
        # Return a single x
        transform = additional_transform_functions[additional_transform](config, dataset, base_transform_steps, normalize, default_normalization)  
    
    if should_rgb_transform:
        transform = apply_rgb_transform(transform)

    return transform


def initialize_bert_transform(config):
    def get_bert_tokenizer(model):
        if model == "bert-base-uncased":
            return BertTokenizerFast.from_pretrained(model)
        elif model == "distilbert-base-uncased":
            return DistilBertTokenizerFast.from_pretrained(model)
        else:
            raise ValueError(f"Model: {model} not recognized.")

    assert "bert" in config.model
    assert config.max_token_length is not None

    tokenizer = get_bert_tokenizer(config.model)

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_token_length,
            return_tensors="pt",
        )
        if config.model == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif config.model == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform


def get_image_base_transform_steps(config, dataset) -> List:
    transform_steps = []

    if dataset.original_resolution is not None and min(
        dataset.original_resolution
    ) != max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))

    if config.target_resolution is not None and config.dataset != "fmow":
        transform_steps.append(transforms.Resize(config.target_resolution))

    return transform_steps


def get_image_resize_and_center_crop_transform_steps(config, dataset) -> List:
    """
    Resizes the image to a slightly larger square then crops the center.
    """
    transform_steps = get_image_resize_transform_steps(config, dataset)
    target_resolution = _get_target_resolution(config, dataset)
    transform_steps.append(
        transforms.CenterCrop(target_resolution),
    )
    return transform_steps


def get_image_resize_transform_steps(config, dataset) -> List:
    """
    Resizes the image to a slightly larger square.
    """
    assert dataset.original_resolution is not None
    assert config.resize_scale is not None
    scaled_resolution = tuple(
        int(res * config.resize_scale) for res in dataset.original_resolution
    )
    return [
        transforms.Resize(scaled_resolution)
    ]


def initialize_poverty_transform(is_training):
    if is_training:
        transforms_ls = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1),
            transforms.ToTensor()]
        rgb_transform = transforms.Compose(transforms_ls)

        def transform_rgb(img):
            # bgr to rgb and back to bgr
            img[:3] = rgb_transform(img[:3][[2,1,0]])[[2,1,0]]
            return img
        transform = transforms.Lambda(lambda x: transform_rgb(x))
        return transform
    else:
        return None


def get_poverty_train_transform_steps() -> List:
    return [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1),
    ]


def apply_rgb_transform(transform):
    def transform_rgb(img):
        # bgr to rgb and then back to bgr
        img[:3] = transform(img[:3][[2, 1, 0]])[[2, 1, 0]]
        return img

    return transforms.Lambda(lambda x: transform_rgb(x))

def add_no_transform(config, dataset, base_transform_steps, normalize, normalization_transform):
    # No stochastic transforms
    transform_steps = copy.deepcopy(base_transform_steps)
    transform_steps.append(transforms.ToTensor())
    if normalize:
        transform_steps.append(normalization_transform)
    return transforms.Compose(transform_steps)

def add_weak_transform(config, dataset, base_transform_steps, normalize, normalization_transform):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    weak_transform_steps = copy.deepcopy(base_transform_steps)
    weak_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution,
            ),
            transforms.ToTensor(),
        ]
    )
    if normalize:
        weak_transform_steps.append(normalization_transform)
    return transforms.Compose(weak_transform_steps)

def add_rand_augment_transform(config, dataset, base_transform_steps, normalize, normalization_transform):
    # Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    target_resolution = _get_target_resolution(config, dataset)
    strong_transform_steps = copy.deepcopy(base_transform_steps)
    strong_transform_steps.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=target_resolution
            ),
            RandAugment(
                n=config.randaugment_n,
                augmentation_pool=FIX_MATCH_AUGMENTATION_POOL,
            ),
            transforms.ToTensor(),
        ]
    )
    if normalize:
        strong_transform_steps.append(normalization_transform)
    return transforms.Compose(strong_transform_steps)

def _get_target_resolution(config, dataset):
    if config.target_resolution is not None:
        return config.target_resolution
    else:
        return dataset.original_resolution


class MultipleTransforms(object):
    """When multiple transformations of the same data need to be returned."""

    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, x):
        return tuple(transform(x) for transform in self.transformations)
