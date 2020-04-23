from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import torch
import torch.optim as optim
import torch.nn as nn


torch.manual_seed(1)
X = torch.Tensor([[1, 0, 0, -1],[0, 1, 0, -1]])
y = torch.Tensor([1, 1, -1, -1])

for i in range(len(y)):
        if y[i] == 0: y[i] = 1
        else: y[i] = -1




im = len(X[0])
w = torch.Tensor([[2, 2]])
b = torch.Tensor([-1])


step_size = 1e-3
num_epochs = 1
minibatch_size = 20
print("X" , X)

for epoch in range(num_epochs):
    inds = [i for i in range(len(X))]
    print("inds ", inds)
    print (range(len(inds)))
    for i in range(len(inds)):
        print("y[inds[i] " , y[inds[i]])
        print("w " , w )
        print("torch.Tensor(X[inds[i]])) " , torch.Tensor(X[inds[i]]))
        print("b " , b)
        L = max(0, 1 - y[inds[i]] * (torch.dot(w, X[inds[i]] - b)))**2
        if L != 0: # if the loss is zero, Pytorch leaves the variables as a float 0.0, so we can't call backward() on it
            L.backward()
            w.data -= step_size * w.grad.data # step
            b.data -= step_size * b.grad.data # step
            w.grad.data.zero_()
            b.grad.data.zero_()

