import torch
import torch.nn as nn
from torchvision import datasets,transforms
from foundations.fashion_class import FashionCNN

train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
model = FashionCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for images,labels in train_loader:
        scores = model(images)
        loss = loss_fn(scores,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch: ", epoch, " last loss: ", loss.item())