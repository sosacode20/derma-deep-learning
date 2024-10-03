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
) -> tuple[ResNet, nn.NLLLoss, torch.optim.Adam]:
    """Get a ResNet50 model with the specified number of output classes.
    Also it returns the loss function and the optimizer"""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, out_features=num_classes),
        nn.LogSoftmax(dim=1), # For NLLLos()
    )
    loss = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model.to("cuda" if torch.cuda.is_available() else "cpu"), loss, optimizer

def get_resnet50_cross_entropy(
    num_classes: int,
) -> tuple[ResNet, nn.CrossEntropyLoss, torch.optim.Adam]:
    """Get a ResNet50 model with the specified number of output classes.
    Also it returns the loss function and the optimizer"""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, out_features=num_classes),
    )
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model.to("cuda" if torch.cuda.is_available() else "cpu"), loss, optimizer


# model, _, _ = get_resnet50_cross_entropy(num_classes=4)
# torchsummary.summary(model, input_data=(3,224,224))