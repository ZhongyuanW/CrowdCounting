# -*- coding: utf-8 -*-
# @Time    : 9/17/19 8:29 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : net.py
# @Software: PyCharm

import torch
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size):

    if kernel_size % 2 == 1:
        padding = int((kernel_size-1)/2)

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding),
        #nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Net(nn.Module):

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __init__(self):
        super(Net, self).__init__()

        self.conv = conv
        self.pooling = nn.MaxPool2d

        self.branch1 = nn.Sequential(
            self.conv(3, 16, 9),
            self.pooling(2),
            self.conv(16, 32, 7),
            self.pooling(2),
            self.conv(32, 16, 7),
            self.conv(16, 8, 7),
        )

        self.branch2 = nn.Sequential(
            self.conv(3, 20, 7),
            self.pooling(2),
            self.conv(20, 40, 5),
            self.pooling(2),
            self.conv(40, 20, 5),
            self.conv(20, 10, 5),
        )

        self.branch3 = nn.Sequential(
            self.conv(3, 24, 5),
            self.pooling(2),
            self.conv(24, 48, 3),
            self.pooling(2),
            self.conv(48, 24, 3),
            self.conv(24, 12, 3),
        )

        self.concat = torch.cat
        self.out = nn.Conv2d(30, 1, 1)
        self.init_params()


    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        x = self.concat((branch1,branch2,branch3),1)
        x = self.out(x)

        return x

if __name__ == "__main__":

    net = Net()

    x = torch.ones((1, 3, 256, 256))
    print(x.size())
    y = net(x)
    print(y.size())
