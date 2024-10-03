from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2
import torchsummary
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch
from torchvision.transforms.v2.functional import InterpolationMode


def get_device():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def get_efficient_net_b2(
    num_classes: int = 4,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-5,
):
    model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(1408),
        nn.Dropout(dropout_rate),
        nn.Linear(1408, num_classes),
    )
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = get_device()
    print(f"The name of the device is {device}")
    return model.to(device=device), loss, optimizer


def get_efficient_net_b2_transformations():
    train_transform = transforms.Compose(
        [
            transforms.Resize(330, interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(
                brightness=(0.9, 1.2),
                contrast=(0.9, 1.2),
                hue=(-0.02, 0.02),
            ),
            transforms.RandomAffine(
                degrees=(-15, 15),
                scale=(1.0, 1.1),
            ),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.CenterCrop(288),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(300, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(288),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    return train_transform, validation_transform


# model, _, _ = get_efficient_net_b2(num_classes=4)
# torchsummary.summary(model.to(device="cpu"), input_data=(3, 288, 288))
