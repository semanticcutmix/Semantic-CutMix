import torch
import torch.nn as nn
import math

class augmentationDotProduct(nn.Module):
    def __init__(self):
        super(augmentationDotProduct, self).__init__()
        self.weights = torch.nn.Parameter(data=(torch.randn(30) / math.sqrt(30)), requires_grad=True)        
    
    def forward(self, x):
        x = torch.matmul(x, self.weights)
        return x

class augmentationEnsembleNet(nn.Module):
    def __init__(self):
        super(augmentationEnsembleNet, self).__init__()
        self.dot = augmentationDotProduct()

    def forward(self, x):
        x = self.dot(x)
        return x

class classEnsembleNet(nn.Module):
    def __init__(self):
        super(classEnsembleNet, self).__init__()
        self.weights = torch.nn.Parameter(data=(torch.randn(3000) / math.sqrt(30)), requires_grad=True)

    def forward(self, x):
        x = x * self.weights
        x = x.reshape(-1,100,30)
        x = torch.sum(x, 2)
        return x


class augEnsembleNet_NetworkAugmentation(nn.Module):
    def __init__(self):
        super(augEnsembleNet_NetworkAugmentation, self).__init__()
        self.weights = torch.nn.Parameter(data=(torch.randn((1,3)) / math.sqrt(3)), requires_grad=True)


    def forward(self, x):
        x = x * self.weights
        x = torch.sum(x, 2)
        return x

class classEnsembleNet_NetworkAugmentation(nn.Module):
    def __init__(self):
        super(classEnsembleNet_NetworkAugmentation, self).__init__()
        self.weights = torch.nn.Parameter(data=(torch.randn((100,3)) / math.sqrt(3)), requires_grad=True)


    def forward(self, x):
        x = x * self.weights
        x = torch.sum(x, 2)
        return x

class classEnsembleFCNet(nn.Module):
    def __init__(self):
        super(classEnsembleFCNet, self).__init__()
        self.fc1 = torch.nn.Linear(300, 100)

    def forward(self, x):
        # print(x.shape)
        x = x.reshape(-1,300)
        x = self.fc1(x)
        return x

