import torch

w=torch.tensor(0.0, requires_grad=True)
x=2.0
target=10.0
lr=0.05

for step in range(20):
    prediction=w*x      #guess
    error=(target-prediction)**2    #error
    error.backward()                #slope ->tells which way to move to shrink the error
    with torch.no_grad():
        w-=(lr*w.grad)             #nudge
    w.grad.zero_()                  #reset
    print(step, w.item(),error.item())