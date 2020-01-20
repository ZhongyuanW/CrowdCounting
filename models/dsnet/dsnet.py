# -*- coding: utf-8 -*-
# @Time    : 10/14/19 12:41 PM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : net.py
# @Software: PyCharm

import torch.nn as nn
from torchvision import models
import torch
import os


def conv(in_channel, out_channel, kernel_size, dilation=1, bn=False):
    #padding = 0
    # if kernel_size % 2 == 1:
    #     padding = int((kernel_size - 1) / 2)
    padding = dilation # maintain the previous size
    if bn:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, dilation=dilation,),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, dilation=dilation,),
            # nn.BatchNorm2d(out_channel, momentum=0.005),
            nn.ReLU(inplace=True)
        )

class DDCB(nn.Module):
    def __init__(self):
        super(DDCB,self).__init__()

        self.phase1 = nn.Sequential(nn.Conv2d(512,256,1),nn.ReLU(inplace=True),conv(256,64,3))
        self.phase2 = nn.Sequential(nn.Conv2d(512+64,256,1),nn.ReLU(inplace=True),conv(256,64,3,2))
        self.phase3 = nn.Sequential(nn.Conv2d(512+64+64,256,1),nn.ReLU(inplace=True),conv(256,64,3,3))
        self.phase4 = conv(512+64+64+64,512,3)

    def forward(self, x):
        x0 = x

        x = self.phase1(x)
        x1 = x
        x = torch.cat((x0,x),1)

        x = self.phase2(x)
        x2 = x
        x = torch.cat((x0,x1,x),1)

        x = self.phase3(x)
        x = torch.cat((x0,x1,x2,x),1)

        x = self.phase4(x)

        return x




class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.front_end = nn.Sequential(conv(3, 64, 3),
                                       conv(64, 64, 3),
                                       nn.MaxPool2d(2, 2),
                                       conv(64, 128, 3),
                                       conv(128, 128, 3),
                                       nn.MaxPool2d(2, 2),
                                       conv(128, 256, 3),
                                       conv(256, 256, 3),
                                       conv(256, 256, 3),
                                       nn.MaxPool2d(2, 2),
                                       conv(256, 512, 3),
                                       conv(512, 512, 3),
                                       conv(512, 512, 3)
                                       )

        self.DDCB1 = DDCB()
        self.DDCB2 = DDCB()
        self.DDCB3 = DDCB()

        self.final = nn.Sequential(conv(512, 128, 3),
                        conv(128, 64, 3),nn.Conv2d(64, 1, 1))

        self.init_param()

    def forward(self, x):
        x = self.front_end(x)
        x0 = x

        x = self.DDCB1(x) + x0
        x1 = x

        x = self.DDCB2(x) + x0 + x1
        x2 = x

        x = self.DDCB3(x) + x0 + x1 + x2

        x = self.final(x)
        return x

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print("loading pretrained vgg16!")
        if os.path.exists("weights/vgg16.pth"):
            print("find pretrained weights!")
            vgg16 = models.vgg16(pretrained=False)
            vgg16_weights = torch.load("weights/vgg16.pth")
            vgg16.load_state_dict(vgg16_weights)
        else:
            vgg16 = models.vgg16(pretrained=True)

        for i in range(len(self.front_end.state_dict().items())):
            list(self.front_end.state_dict().items())[i][1].data[:] = list(vgg16.state_dict().items())[i][1].data[:]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    net = Net()
    #print(net.front_end.state_dict())
    x = torch.ones((1, 3, 256, 256))
    print(x.size())
    y = net(x)
    print(y.size())