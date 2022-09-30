import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ToTensor 
import torchvision.transforms as T
import os
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt

#DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' # if your machine supports cuda 
DEVICE = 'cpu'

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,down=True , act = 'relu', use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='reflect')
            if down
            else nn.ConvTranspose2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=False),

            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
      x = self.conv(x)
      return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self,in_channels=3,features=64):
      super().__init__()
      #Down sampling
      self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels,features,kernel_size = 4 , stride = 2 , padding= 1 , bias=False,padding_mode='reflect'),
            nn.LeakyReLU(0.2)   
      ) # [1 ,3, 256 , 256]  -> [1, 64 , 128, 128]
      
      self.down1 = Block(features,features*2,down=True,act='leaky',use_dropout=False) #[1,64,128,128] -> [1,128,64,64]
      self.down2 = Block(features*2,features*4,down=True,act='leaky',use_dropout=False) #[1,128,64,64] ->[1,256,32,32]
      self.down3 = Block(features*4,features*8,down=True,act='leaky',use_dropout=False) # [1,256,32,32] ->[1,512,16,16]
      self.down4 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False) # [1,512,16,16] --> [1,512,8,8] 
      self.down5 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False) # [1,512,8,8] --> [1,512,4,4]
      self.down6 = Block(features*8,features*8,down=True,act='leaky',use_dropout=False) # [1,512,4,4] --> [1,512,2,2]

      self.bottleneck  = nn.Sequential(
          nn.Conv2d(features*8,features*8,4,2,1,padding_mode = 'reflect'),
          nn.ReLU()
      ) #[1, 512, 2 , 2] --> [1, 512, 1, 1]

      #upSampling 
      # here were are doubling the input features because apply skip+concatination
      self.up1 = Block(features*8,features*8,down=False,act='relu', use_dropout=True) #[1, 512,1,1] --> [1, 512 , 2 ,2]
      self.up2 = Block(features*8*2,features*8,down=False,act='relu', use_dropout=True) #[1,512*2 , 2 ,2] --> [1 , 512 , 4, 4]
      self.up3 = Block(features*8*2,features*8,down=False,act='relu', use_dropout=True) # [1 , 512*2, 4,4] --> [1, 512, 8, 8] 
      self.up4 = Block(features*8*2,features*8,down=False,act='relu', use_dropout=True) # [1, 512*2, 8, 8] --> [1, 512, 16, 16]
      self.up5 = Block(features*8*2,features*4,down=False,act='relu', use_dropout=True) # [1, 512*2, 16, 16] --> [1, 256, 32 ,32]
      self.up6 = Block(features*4*2,features*2,down=False,act='relu', use_dropout=True) # [1, 256 *2 , 32, 32 ]  --> [1 ,128 , 64,64]
      self.up7 = Block(features*2*2,features,down=False,act='relu', use_dropout=True)   # [1,128*2 , 64, 64] --> [1,64, 128, 128]

      self.final_up  = nn.Sequential(
          
                  nn.ConvTranspose2d(features*2,in_channels,kernel_size=4,stride=2,padding=1),
                  nn.Tanh(), # this ensure that our pix value ranges from -1 to 1
      ) #[1,64*64, 128 , 128]  --> [1, 3, 256 , 256] 

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1,d7],1))
        up3 = self.up3(torch.cat([up2,d6],1))
        up4 = self.up4(torch.cat([up3,d5],1))
        up5 = self.up5(torch.cat([up4,d4],1))
        up6 = self.up6(torch.cat([up5,d3],1))
        up7 = self.up7(torch.cat([up6,d2],1))
        return self.final_up(torch.cat([up7,d1],1))


def get_image_as_dl(file_path):
    # transformations---------
 
    transform = A.Compose(
        [#T.ColorJitter(),
         A.Normalize(mean=[0.5,0.5,0.5] , std =[0.5,0.5,0.5], max_pixel_value=255.),
         ToTensorV2() #[height, width , numChannel]  --> [numChannel,width, height]
        ]
    )
    
    resize_image = T.Resize((256,256))
    #-----------------------
	
    input_img = np.array(Image.open(file_path).convert('RGB'))
    input_img = transform(image=input_img)["image"]

    #resizing
    input_img = resize_image(input_img)
    
    return (input_img).unsqueeze(0)
    
    
def show_image(image):
  plt.imshow(image.squeeze(0).permute(1,2,0)*0.5+0.5)
  plt.axis('off')
  input("enter")
  
  

if __name__ == '__main__':
	#Input image pre-processing

    image = get_image_as_dl('test/inputs/9.jpg')


    # Loading model

    checkpoint = torch.load("model/gen.pth.tar", map_location=torch.device('cpu'))
    model = Generator()
    model.load_state_dict(checkpoint['state_dict'])
    

    # #prediction / generate image
    photo = model(image.to('cpu'))    

    save_image(photo.squeeze(0)*0.5+0.5,"test/outputs/gen_9.png")


	
