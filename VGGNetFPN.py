import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

#######################################################################################################################
##                                              FPN Implementation                                                   ##           
#######################################################################################################################
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3,padding=1),nn.BatchNorm2d(mid_channels),nn.ReLU(inplace=True),nn.Conv2d(mid_channels, out_channels, kernel_size=3,padding=1),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))

    def forward(self,x):
       return self.double_conv(x)

class FPNVGGNet(nn.Module):
    def __init__(self,DoubleConv):
        super(FPNVGGNet,self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        #Bottom-up layers
        self.layer1 = DoubleConv(64, 128)
        self.layer2 = DoubleConv(128, 256)
        self.layer3 = DoubleConv(256, 512)
        self.layer4 = DoubleConv(512, 1024)

        #Top layer
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1,stride=1,padding=0)

        #Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1)

        #Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1,stride=1,padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1,stride=1,padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1,stride=1,padding=0)


    def _upsample_add(self,x,y):
        """
        Upsample and add two feature maps.

        Args:
        x:Top feature map to be sampled
        y:Lateral feature map

        Returns:
        Added feature map
        """

        _,_,H,W = y.size()
        return F.upsample(x,size=(H,W),mode='bilinear') + y

    def forward(self,x):
        #Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1,kernel_size=3,stride=2,padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        #Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        print(p2,p3,p4,p5)
        #Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2,p3,p4,p5

#Testing
if __name__ == '__main__':
    a = FPNVGGNet(DoubleConv)
    print(a)
    