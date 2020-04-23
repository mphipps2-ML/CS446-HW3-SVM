import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

test1 = torch.Tensor([[2,0,1,3]])
test2 = torch.Tensor([1,1,-1,-1])
print("test 1*2 " , torch.matmul(test1,test2)) #equals -2



torch.manual_seed(1)
X = torch.Tensor([[1, 0, 0, -1],[0, 1, 0, -1]])
y = torch.Tensor([1, 1, -1, -1])
alpha = 0.001
C = 1

class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(2,1, bias=True)
    
    def forward(self, X):
        return self.fc1(X)

net = ShallowNet()
print(net)

print(net(torch.transpose(X,0,1)).squeeze())

optimizer = optim.SGD(net.parameters(), lr=alpha, weight_decay=0)
optimizer.zero_grad()

params = list(net.parameters())
params[0].data = torch.Tensor([[2, 2]])
params[1].data = torch.Tensor([-1])

farr = []



print("y ", y )
print("w ", params[0].data)
print("X ", X)
#print("y*params[0].data*X", y*params[0].data*X)

print("y*params[1].data",(y*params[1].data))
for iter in range(10000):
    inds = [i for i in range(len(X))]
    loss = 0.
    for i in range(len(inds)):
        
        if iter==0:
            print(1 - y*net(torch.transpose(X,0,1)).squeeze())
        ##############################
        ## Complete this single line which is our cost function
        ## Dimensions: loss (scalar)
        ##############################
        #    loss = C/2* + torch.sum(torch.max(0,1-(y*params[0].data,X) - (y*params[1].data)),)1
        loss = loss + C/2 * torch.max(0,1-(y*( torch.dot(params[0].data,torch.Tensor(X[inds[i]])) - (y*params[1].data))))
    
        #    loss = C/2* + torch.sum(torch.clamp(1-(y*params[0].data,X) - (y*params[1].data),min=0))
        #        loss = torch.max(1-(y*params[0].data,X) - (y*params[1].data),min=0))
    
    loss.backward()
    gn = 0
    for f in net.parameters():
        if iter==0:
            print("Test")
            print(f.grad)
        gn = gn + torch.norm(f.grad)
    print("Iter: %d; Loss: %f; ||g||: %f" % (iter, loss, gn))
    optimizer.step()
    optimizer.zero_grad()

    farr.append(loss.item())

    for f in net.parameters():
        print(f)

plt.plot(farr)
plt.show()
