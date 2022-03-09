from torch import nn
import torch
import torchvision
from matplotlib import pyplot as plt
import torch.nn.functional as F


class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=5,device=device)
    self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=7,device=device)
    self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5,device=device)
    self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=7,device=device)
    self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5,device=device)
    self.activation = torch.nn.ReLU()

  def forward(self,x):
    x = self.activation(self.deconv1(x))
    x = self.activation(self.deconv2(x))
    x = self.activation(self.deconv3(x))
    x = self.activation(self.deconv4(x))
    x = self.deconv5(x)
    return x




class Discriminator(nn.Module):
  #insize = 4X4x1
  #outsize = 28x28x1
  def __init__(self,drop):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5,device=device)
    self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3,device=device)
    self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3,device=device)      
    self.pool = nn.AvgPool2d(kernel_size=2)
    self.flat = torch.nn.Flatten()
    self.activation = torch.nn.ReLU()
    self.dense1 = nn.Linear(in_features=576, out_features=32,device=device)
    self.dense2 = nn.Linear(in_features=32, out_features=8,device=device)
    self.dense3 = nn.Linear(in_features=8, out_features=1,device=device)
    self.sig = nn.Sigmoid()
    self.drop = torch.nn.Dropout(p=drop)


  def forward(self,x):
    
    x = self.drop(self.activation(self.pool(self.conv1(x))))
    x = self.drop(self.activation(self.pool(self.conv2(x))))
    x = self.drop(self.activation(self.conv3(x)))
    x = self.flat(x)
    x = self.drop(self.activation(self.dense1(x)))
    x = self.drop(self.activation(self.dense2(x)))
    x = self.dense3(x)

    return x



class GAN(nn.Module):
  def __init__(self,drop):
    mnist_dim = 784
    z_dim = 100
    super().__init__()
    self.discriminator = Discriminator(mnist_dim).to(device)
    self.generator = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
    self.optd = torch.optim.Adam(self.discriminator.parameters(),lr=2e-4)#,betas=(0.5, 0.999))
    self.optg = torch.optim.Adam(self.generator.parameters(),lr=2e-4)#,betas=(0.5, 0.999))    
    self.loss = torch.nn.BCELoss()

  def forward(self,x):
    x = self.generator(x)
    x = self.discriminator(x)
    return x

  
  def train(self,dataloader,batch_size,epochs, k):
    
    num_epochs = 0
    iteration = 0
    dataset = iter(dataloader)
    loss_d_avg = 0
    loss_g_avg = 0
    while(1):  
        iteration += 1
        try:
          x_pos = next(dataset)[0].to(device)
          x_pos = x_pos.view((x_pos.shape[0],784))
        except StopIteration:
          print(num_epochs)        
          print("dis loss")
          print(loss_d_avg)          
          print("gen loss")
          print(loss_g_avg)

          
          img = self.generator(torch.randn(size=(1,100),device=device)).view((28,28))
          plt.imshow(img.detach().cpu().numpy(), cmap="gray")
          plt.show()

          num_epochs += 1
          if num_epochs >= epochs:
            return
          dataset = iter(dataloader)

        
        
        self.optd.zero_grad() 
        y_pos = torch.ones(size = (x_pos.shape[0],1)).to(device)
        y_pred_real = self.discriminator(x_pos)
        loss_d_real = self.loss(y_pred_real,y_pos)
        noise = torch.randn(size=(batch_size,100),device=device)
        x_neg = self.generator(noise)
        y_neg = torch.zeros(size = (x_neg.shape[0],1)).to(device)
        y_pred_fake = self.discriminator(x_neg)
        loss_d_fake = self.loss(y_pred_fake,y_neg)
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        loss_d_avg = (loss_d_avg*(iteration-1) + loss_d.item())/iteration
        self.optd.step()



        X = torch.randn(size=(batch_size,100),device=device)
        y = torch.ones(size = (X.shape[0],1)).to(device)
        self.optg.zero_grad()
        y_pred = self.forward(X)
        loss_g = self.loss(y_pred, y)
        loss_g.backward()
        loss_g_avg = (loss_g_avg*(iteration-1) + loss_g.item())/iteration
        self.optg.step()