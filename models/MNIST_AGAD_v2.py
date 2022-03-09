from torch import nn
import torch
import torchvision
from matplotlib import pyplot as plt
import torch.nn.functional as F



class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.dense1 = nn.Linear(in_features=64, out_features=1024,device=device)
    self.dense2 = nn.Linear(in_features=1024, out_features=1024,device=device)
    self.dense3 = nn.Linear(in_features=1024, out_features=784,device=device)
    self.activation = torch.nn.ReLU()
    self.tanh = torch.nn.Tanh()


  def forward(self,x):
    x = self.activation(self.dense1(x))
    x = self.activation(self.dense2(x))
    x = torch.tanh(self.dense3(x))
    return x#.view((x.shape[0],1,28,28))



class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.dense1 = nn.Linear(in_features=784, out_features=128,device=device)
    self.dense2 = nn.Linear(in_features=128, out_features=64,device=device)
    self.dense3 = nn.Linear(in_features=64, out_features=1,device=device)
    # self.activation = torch.nn.ReLU()
    self.activation =  torch.nn.LeakyReLU(negative_slope=0.01)

  def forward(self,x):
    x = self.activation(self.dense1(x))
    x = self.activation(self.dense2(x))
    x = self.dense3(x)
    return x
