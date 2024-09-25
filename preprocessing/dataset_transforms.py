import math
from typing import *
import torchvision.transforms.v2 as transforms
from .dull_razor import RandomDullRazor
import torch

def get_classification_train_transforms_v2(
    image_size: tuple[int, int] = (224, 224), dull_razor_probability: float = 0.5
):
    """The transformations to use with the classifier when training"""
    return transforms.Compose(
        [
            transforms.Resize(
                image_size,  # Default is (512,512)
                antialias=True,
            ),
            RandomDullRazor.from_image_size(image_size, probability=dull_razor_probability),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ColorJitter(
                brightness=(0.8, 1.2),  # Rango de valores: 0.0 a 1.0 o un solo valor
                contrast=(0.8, 1.3),  # Rango de valores: 0.0 a 1.0 o un solo valor
                saturation=(0.8, 1.3),  # Rango de valores: 0.0 a 1.0 o un solo valor
                hue=(-0.02, 0.02),  # Rango de valores: -0.5 a 0.5
            ),
            transforms.RandomAffine(
                degrees=(-180, 180),
                scale=(1.0, 1.2),
            ),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
        ]
    )

def get_classification_evaluation_transform_v2(image_size: tuple[int, int] = (224, 224),):
    """The transformations to use with the classifier when inferring or validating"""
    return transforms.Compose(
        [
            transforms.Resize(
                image_size,
                antialias=True,
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )


