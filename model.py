# model.py
import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=4, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    return model
