import torch
import torch.nn as nn
from torchvision import datasets,transforms

data=datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
loader=torch.utils.data.DataLoader(data,batch_size=64,shuffle=True)
images,labels=next(iter(loader))
print(images.shape)
print(labels[:10])

class FashionCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,8,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(8,16,kernel_size=3,padding=1)
        self.pool= nn.MaxPool2d(2)
        self.fc=nn.Linear(1,10)

    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)))
        print("after block 1: ", x.shape)
        x=self.pool(torch.relu(self.conv2(x)))
        print("after block 2: ", x.shape)
        x=x.flatten(start_dim=1)
        print("after flatten: ", x.shape)
        x=self.fc(x)
        print("output: ", x.shape)
        return x

model=FashionCNN()
model(images)