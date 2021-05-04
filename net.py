import torch.nn as nn
import torch
import torch.nn.functional as F

#########################################################################################################
##                                       VGGNet Implementation                                         ## 
#########################################################################################################

VGG16 = [32,32,'M',64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M']

class VGG_Net(nn.Module):
    def __init__(self,in_channels):
        super(VGG_Net,self).__init__()
        self.in_channels = in_channels
        #self.conv_layers = create_conv_layers(VGG16)

    def create_conv_layers(self,architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if(type(x)) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = 3,stride=1,padding=1),nn.BatchNorm2d(x),nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]

            return nn.Sequential(*layers)

    def forward(self,x):
        x = create_conv_layers(VGG16)
        return x

#######################################################################################################################
##                                              FPN Implementation                                                   ##           
#######################################################################################################################
class BasicBlock(nn.Module):
    expansion = 4
    def  __init__(self,in_channels,out_channels,identity_downsample=None,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels,out_channels * self.expansion,kernel_size=1,stride=stride,padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self,x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        print(x.shape())
        x = self.relu(x)

        return x

class FPNRes(nn.Module):
    def __init__(self,BasicBlock,num_blocks):
        super(FPNRes,self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        #Bottom-up layers
        self.layer1 = self._make_layer(BasicBlock,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock,  128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock,  256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock,  512, num_blocks[3], stride=2)

        #Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1,stride=1,padding=0)

        #Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3,stride=1,padding=1)

        #Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1,stride=1,padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1,stride=1,padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1,stride=1,padding=0)

    def _make_layer(self,BasicBlock,planes,num_blocks,stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes,planes,stride))
            self.in_planes = planes * BasicBlock.expansion
        
        return nn.Sequential(*layers)

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

        #Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2,p3,p4,p5



















