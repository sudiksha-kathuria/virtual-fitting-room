import torch

a=torch.tensor([[1.0,2.0],[3.0,4.0]])
print(a.shape)
print(a*2)

x=torch.tensor(3.0,requires_grad=True)
y=x**2
y.backward()
print(x.grad)
print(y)