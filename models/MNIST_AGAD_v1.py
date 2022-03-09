from torch import nn
import torch
import torchvision
from matplotlib import pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))



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