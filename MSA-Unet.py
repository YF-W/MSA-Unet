import torchvision.transforms.functional as TF
# import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


"""
MSA-unet

Author: Xiaoyan Guo
Affiliation: Chengdu University
For academic and research purposes only, not for commercial use.

"""

class Rat(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        N, C, H, W = x.shape
        input_x=x.view(N,C,H*W)#N,C,H*W
        input_x = input_x.unsqueeze(1)#N,1,C,H*w
        context_mask= self.conv1(x)  #N,1,H,W
        context_mask=context_mask.view(N,1,H*W)#N,1,H*W
        context_mask=self.softmax(context_mask)#N,1,H*W
        context_mask=context_mask.view(N,1,H*W,1)#N,1,H*W,1
        out = torch.matmul(input_x, context_mask)#N,1,C,1
        out = out.view(N, C, 1, 1)
        return out



class Cat1(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.l1 = nn.Linear(in_channels,in_channels//8)
        self.l2 = nn.Linear(in_channels//8,in_channels//16)
        self.l3 = nn.Linear(in_channels//16,in_channels//8)
        self.l4 = nn.Linear(in_channels//8, in_channels)
        # self.layer=nn.LayerNorm([3, 1, 1])
        self.relu=nn.ReLU(inplace=True)
        self.rat=Rat(in_channels)

    def forward(self, x):
        b,c,_,_ =x.shape
        out = self.rat(x).view(b,c)
        out = self.l1(out)
        out = self.l2(out)
        # out = self.layer(out)
        out = self.relu(out)
        out = self.l3(out)
        out =self.l4(out).view(b,c,1,1)
        return out



class Cat2(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels*8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels*8,in_channels*16,kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels*16,in_channels*8,kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels*8, in_channels, kernel_size=1)
        # self.layer=nn.LayerNorm([3, 1, 1])
        self.relu=nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        self.rat=Rat(in_channels)
        self.cat1=Cat1(in_channels)

    def forward(self, x):
        out = self.rat(x)
        out = self.conv1(out)
        out = self.conv2(out)
        # out = self.layer(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.sigmoid(out)
        out = out + self.cat1(x)
        return out



class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention, self).__init__()
        self.conv=nn.Conv2d(in_channels,1,1)
        self.avg1=nn.AdaptiveAvgPool2d((None,1))
        self.avg2=nn.AdaptiveAvgPool2d((1,None))
        self.sigmoid=nn.Sigmoid()
        self.cat2=Cat2(in_channels)
    def forward(self,x):
        out=self.conv(x)
        tat1=self.avg1(out)
        tat2=self.avg2(out)
        tat=tat1*tat2
        out=self.sigmoid(tat)
        out=out+self.cat2(x)*x
        return out



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):

        return self.conv(x)



class DoubleConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)



class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Depthwise, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,  bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

     return self.conv(x)



class MSA(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(MSA, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = Depthwise(features[-1], features[-1] * 2)
        self.finalconv = nn.Conv2d(1984, out_channels, kernel_size=1)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        # self.lat=nn.ModuleList()
        for i, feature in enumerate(features):
            if i > 0:
                in_channels = in_channels + features[i - 1]
                self.downs.append(DoubleConv(in_channels, feature))
            elif i ==0 :
                self.downs.append(DoubleConv1(in_channels, feature))

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(Attention(feature))
            self.ups.append(DoubleConv(feature * 2, feature))

    def forward(self, x):
        skip_connections = []
        d = []
        wait_concat = []
        wait_concat.append(x)

        for down in self.downs:
            if len(wait_concat)!=1:
                for i in range(len(wait_concat)-1):
                    y=wait_concat[i]
                    if y.shape!=x.shape:
                        y=TF.resize(y,size=x.shape[2:])
                    x=torch.cat([x,y],dim=1)
            x=down(x)
            skip_connections.append(x)
            x=self.pool(x)
            wait_concat.append(x)

        x = self.bottleneck(x)
        d.append(x)

        skip_connections = skip_connections[::-1]


        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 3]
            if idx > 6:
                skip_connection = self.ups[idx+1](skip_connection)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 2](concat_skip)
            d.append(x)
        for idx in range(0,len(d)-1):
            if d[idx].shape!=d[4].shape:
                d[idx]=TF.resize(d[idx],size=d[4].shape[2:])
                d[4]=torch.cat((d[4],d[idx]),dim=1)
        x=d[4]
        return self.finalconv(x)


x = torch.randn(3, 3, 512, 512)
model = MSA(in_channels=3,out_channels=1)
preds = model(x)
print(preds.shape)
print(x.shape)
