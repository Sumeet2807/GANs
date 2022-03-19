from torch import nn
import torch
import torchvision
from matplotlib import pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
  def __init__(self,nz,ngf,nc):
    super().__init__()
    self.fwd = nn.Sequential(
      nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*4) x 7 x 7
      nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. (ngf*2) x 14 x 14
      nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (ngf*2) x 28 x 28
    )

  def forward(self,x):
    return self.fwd(x)




class Discriminator(nn.Module):
    def __init__(self,ndf,nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class GAN(nn.Module):
  def __init__(self,nz,ngf):
    super().__init__()
    self.discriminator = Discriminator(ngf,1).to(device)
    self.generator = Generator(nz,ngf,1).to(device)
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
        except StopIteration:
          print(num_epochs)        
          print("dis loss")
          print(loss_d_avg)          
          print("gen loss")
          print(loss_g_avg)

          
          img = self.generator(torch.randn(size=(1,100,1,1),device=device)).view((28,28))
          plt.imshow(img.detach().cpu().numpy(), cmap="gray")
          plt.show()

          num_epochs += 1
          if num_epochs >= epochs:
            return
          dataset = iter(dataloader)

        
        
        self.optd.zero_grad() 
        y_pos = torch.ones(size = (x_pos.shape[0],1)).to(device)
        y_pred_real = self.discriminator(x_pos)
        loss_d_real = self.loss(torch.flatten(y_pred_real,1),y_pos)
        noise = torch.randn(size=(batch_size,100,1,1),device=device)
        x_neg = self.generator(noise)
        y_neg = torch.zeros(size = (x_neg.shape[0],1)).to(device)
        y_pred_fake = self.discriminator(x_neg)
        loss_d_fake = self.loss(torch.flatten(y_pred_fake,1),y_neg)
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        loss_d_avg = (loss_d_avg*(iteration-1) + loss_d.item())/iteration
        self.optd.step()



        X = torch.randn(size=(batch_size,100,1,1),device=device)
        y = torch.ones(size = (X.shape[0],1)).to(device)
        self.optg.zero_grad()
        y_pred = self.forward(X)
        loss_g = self.loss(torch.flatten(y_pred,1), y)
        loss_g.backward()
        loss_g_avg = (loss_g_avg*(iteration-1) + loss_g.item())/iteration
        self.optg.step()