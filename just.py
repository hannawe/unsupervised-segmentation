

# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys

import numpy as np
import torch.nn.init

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False,
                    help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                    help='number of channels')
# parser.add_argument('--maxIter', metavar='T', default=500, type=int,
parser.add_argument('--maxIter', metavar='T', default=5, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=2, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float,
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float,
                    help='step size for scribble loss')
args = parser.parse_args()

class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()

        # w x h x c -> (w-k) x (h-k) x c' (RGB summed up)
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1) # number of channels in the input image
        self.bn1 = nn.BatchNorm2d(args.nChannel) # (w-3) x (h-3) x c'
        self.conv2 = nn.ModuleList() # Holds submodules in a list
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv - 1):
            self.conv2.append(nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1)) # padding 1 keeps the image size
            self.bn2.append(nn.BatchNorm2d(args.nChannel))
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0) # padding =0 but kernel size = 1 --> equivalent to FC layer
        # used for adjusting number of channels btwn layers and complexity
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    # nConv = 2 by default
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(args.nConv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x