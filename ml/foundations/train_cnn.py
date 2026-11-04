import torch
import torch.nn as nn
from torchvision import datasets,transforms
from foundations.fashion_class import FashionCNN

train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
model = FashionCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def accuracy(loader):
    correct=0
    total=0
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images)
            guesses = scores.argmax(dim=1)
            correct += (guesses==labels).sum().item()
            total += len(labels)
    return correct/total

for epoch in range(15):
    for images,labels in train_loader:
        scores = model(images)
        loss = loss_fn(scores,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch: ", epoch, " train accuracy: ", accuracy(train_loader), " test accuracy: ", accuracy(test_loader))

torch.save(model.state_dict(), "fashion_cnn.pt")