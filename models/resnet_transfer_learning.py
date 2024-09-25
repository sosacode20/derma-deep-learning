from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    ResNet,
)
import torch.nn as nn
import torch
import torchsummary


def get_resnet50(
    num_classes: int,
) -> tuple[ResNet, nn.CrossEntropyLoss, torch.optim.SGD]:
    """Get a ResNet50 model with the specified number of output classes.
    Also it returns the loss function and the optimizer"""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2048, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, out_features=num_classes),
    )
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    return model.to("cuda" if torch.cuda.is_available() else "cpu"), loss, optimizer
