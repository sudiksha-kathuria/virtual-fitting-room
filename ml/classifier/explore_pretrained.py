import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print(model.fc)
print("total weights:", sum(p.numel() for p in model.parameters()))

for p in model.parameters():
    p.requires_grad = False         # BLANK 1: freeze everything

model.fc = nn.Linear(512, 5)        # BLANK 2: new final layer for 5 categories

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable weights:", trainable)