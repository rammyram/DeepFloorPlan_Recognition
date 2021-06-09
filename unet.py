import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))

    def forward(self,x):
        return self.conv(x)

    
class UNet(nn.Module):
    def __init__(self,n_classes,in_channels=3,features=[64,128,256]):
        super(UNet,self).__init__()
        self.n_classes = n_classes
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        #Downpart of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature

        #Uppart of UNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2,feature,kernel_size=2,stride=2))
            self.ups.append(DoubleConv(feature * 2,feature))

        self.bottleneck = DoubleConv(features[-1],features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0],n_classes,kernel_size=1)
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(in_channels),nn.ReLU())

    def forward(self,x):
        skip_connections = []
        #x = self.conv(x)
        for down in self.downs:
            #print(x.shape)
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.ups),2):
            #print(x.shape)
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[idx + 1](concat_skip)
            
        return self.final_conv(x)
   
"""
if __name__ == '__main__':
    net = UNet(n_classes=3)
    summary(net,(1,600,600))
"""