import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print(model.fc)
print("total weights:", sum(p.numel() for p in model.parameters()))

for p in model.parameters():
    p.requires_grad = False         

model.fc = nn.Linear(512, 5)        

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable weights:", trainable)