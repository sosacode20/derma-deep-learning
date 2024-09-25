import torchvision.models as models
import torch.nn as nn


# def get_vgg16(num_classes):
#     model = models.vgg16(pretrained=True)
#     # Freeze parameters so we don't backprop through them
#     for param in model.parameters():
#         param.requires_grad = False
#     model.avgpool = nn.Sequential(
#         nn.Conv2d(512, 512, kernel_size=3), nn.MaxPool2d(2), nn.ReLU(), nn.Flatten()
#     )
#     model.classifier = nn.Sequential(
#         nn.Linear(2048, 512),
#         nn.ReLU(),
#         nn.Dropout(0.4),
#         nn.Linear(512, 128),
#         nn.ReLU(),
#         nn.Dropout(0.4),
#         nn.Linear(128, 64),
#         nn.ReLU(),

#         nn.Linear(64, num_classes),
#         nn.Softmax(),
#     )

