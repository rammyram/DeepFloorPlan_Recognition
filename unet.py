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
        #self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self,x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.ups[idx + 1](concat_skip)
        
        x = self.final_conv(x)
        #print(x.shape)
        return x
   

if __name__ == '__main__':
    net = UNet(n_classes=3)

    if torch.cuda.device_count() > 0:
        summary(net.cuda(), (3,600,600))
    else:
        summary(net, (3,600,600))
