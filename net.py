import torch.nn as nn
import torch

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

############################################################################################################
##                                        ResNet Implementation                                           ##  
############################################################################################################
class BasicBlock(nn.Module):
    def  __init__(self,in_channels,out_channels,identity_downsample=None,stride=1):
        super(BasicBlock,self).__init__()
        self.expansion = 4
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

class ResNet(nn.Module):
    def __init__(self,block,layers,image_channels): #layers: Tells us how many times we reuse the resnet blocks. [3,4,6,3] for ResNet50
        super(ResNet,self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels,64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        #ResNet layers
        self.layer1 = self._make_layer(block,layers[0],out_channels=64,stride=1)
        self.layer2 = self._make_layer(block,layers[1],out_channels=128,stride=2)
        self.layer3 = self._make_layer(block,layers[2],out_channels=256,stride=2)
        self.layer4 = self._make_layer(block,layers[3],out_channels=512,stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x

    def _make_layer(self,BasicBlock,num_residual_blocks,out_channels,stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,out_channels * 4,kernel_size=1,stride=stride),nn.BatchNorm2d(out_channels * 4))
        
        layers.append(BasicBlock(self.in_channels,out_channels,identity_downsample,stride))
        self.in_channels = out_channels * 4 #256

        for i in range(num_residual_blocks - 1):
            layers.append(BasicBlock(self.in_channels,out_channels))

        return nn.Sequential(*layers)

#######################################################################################################################
##                                              FPN Implementation                                                   ##           
#######################################################################################################################



















def ResNet18(img_channels=3):
    return ResNet(BasicBlock,[2,2,2,2],img_channels)


def ResNet50(img_channels=3):
    return ResNet(BasicBlock,[3,4,6,3],img_channels)

def ResNet101(img_channels=3):
    return ResNet(BasicBlock,[3,4,23,3],img_channels)


def test():
    net = ResNet50()
    x = torch.randn(2,3,600,600)
    y = net(x).to('cuda')
    #print(y.shape)

