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

import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"

import random

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

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, r'C:\\Users\\admin\\Desktop\\H.We\\ML-local\\pytorch-unsupervised-segmentation-tip')

# import sys
# f = open("log.out", 'w')
# sys.stdout = f
# sys.stderr = f

# cd C:\Users\admin\Desktop\H.We\ML-local\pytorch-unsupervised-segmentation-tip

# CNN model
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

#######
path = r'C:\Users\admin\Desktop\H.We\ML-local\pytorch-unsupervised-segmentation-tip'
file = path+r'\test.jpg'
im = cv2.imread(file,1)
mask = cv2.imread((path+r'\test_scribble.png'),1) # --> (326, 334, 3)
mask = cv2.imread((path+r'\test_scribble.png'),-1) # --> (326, 334, 4) # there is additional alpha channel
#######

#######
path = r'C:\Users\admin\Desktop\H.We\ML-local\pytorch-unsupervised-segmentation-tip'
file1 = path+r'\PASCAL_VOC_2012\2007_005915.jpg'
im1 = cv2.imread(file1,1)
mask1 = cv2.imread((path+r'\PASCAL_VOC_2012\2007_005915_scribble.png'),1) # --> (375, 500, 3)
mask1 = cv2.imread((path+r'\PASCAL_VOC_2012\2007_005915_scribble.png'),-1) # --> (375, 500)
#######



# load image
im = cv2.imread(args.input) # default color=1 gray=0
data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
if use_cuda:
    data = data.cuda()
data = Variable(data)

# load scribble
if args.scribble:
    mask = cv2.imread(args.input.replace('.' + args.input.split('.')[-1], '_scribble.png'), -1) # read unchanged including alpha channel

    if len(mask.shape) > 2:
        mask = cv2.imread(args.input.replace('.' + args.input.split('.')[-1], '_scribble.png'), 0)

    mask = mask.reshape(-1)
    mask_inds = np.unique(mask)
    mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
    inds_sim = torch.from_numpy(np.where(mask == 255)[0])
    inds_scr = torch.from_numpy(np.where(mask != 255)[0]) # scribble indices
    # target_scr = torch.from_numpy(mask.astype(np.int)) # this solved the problem for 'Int'

    target_scr = torch.from_numpy(mask)
    if use_cuda:
        inds_sim = inds_sim.cuda()
        inds_scr = inds_scr.cuda()
        target_scr = target_scr.cuda()
    target_scr = Variable(target_scr)
    # set minLabels
    args.minLabels = len(mask_inds) # which means this varies ? probably

print('0')

# train
model = MyNet(data.size(1))
if use_cuda:
    model.cuda()
model.train()

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# scribble loss definition
loss_fn_scr = torch.nn.CrossEntropyLoss()

# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average=True)
loss_hpz = torch.nn.L1Loss(size_average=True)

HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], args.nChannel)
HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, args.nChannel)
if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255, size=(100, 3))

print('1')
for batch_idx in range(args.maxIter):

    print('batch ', batch_idx)

    # forwarding
    optimizer.zero_grad()
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel) # tensor shape: [raveled;  nchannel]

    # print(output)
    print('output.shape', output.shape)

    # contiguous makes a copy of tensor such that the memory layout/order of its elements is the same as if it is created
    # a.T will be Fortran contiguous whereas a is C contiguous, and if a.T.reshape(same dim) will cause an error
    # since it is not possible to have inconstant stride length (jumps to memory address to access data value)
    # need to make a copy and do the operation on that copy
    # basically the same operation as: arr2_copy = arr2(=arr.T).copy(), arr2_copy.shape = (12,) --> will not cause error!
    # view does not make a copy of the original tensor. It changes the dimensional interpretation (striding) on the original data

    print('1-0')

    outputHP = output.reshape((im.shape[0], im.shape[1], args.nChannel)) # tensor shape: [width length ; nchannel]

    # print(outputHP)
    print('outputHP shape: ', outputHP.shape)

    HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
    HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
    lhpy = loss_hpy(HPy, HPy_target)
    lhpz = loss_hpz(HPz, HPz_target)

    print('HPy, HPz: ', HPy.shape, HPz.shape)
    print('HPy_target, HPz_target: ', HPy_target.shape, HPz_target.shape)

    print('1-1')

    ignore, target = torch.max(output, 1) # Returns the maximum value of all elements in the input tensor =1--> max of rows
    im_target = target.data.cpu().numpy()

    print('im_target shape: ', im_target.shape)

    nLabels = len(np.unique(im_target))

    print('nLabels: ', nLabels)
    if args.visualize:
        im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
        cv2.imshow("output", im_target_rgb)
        cv2.waitKey(10)

    print('1-2')



    print(output.shape)
    print(target.shape)
    print(inds_sim.shape)

    # print('output[inds_sim]: ', output[inds_sim])

    print('output[inds_sim].shape: ', output[inds_sim].shape)

    # print('target[inds_sim]: ', target[inds_sim])
    print('target[inds_sim].shape: ', target[inds_sim].shape)

    print('loss fn: ', loss_fn(output[inds_sim], target[inds_sim]))
    # print(inds_sim)
    # print(output[inds_sim])
    # print(target[inds_sim])
    print('scr loss')

    # passed

    # print(output[inds_scr])
    # print(target[inds_scr])

    print('output[inds_scr].shape: ', output[inds_scr].shape)
    print('target[inds_scr].shape: ', target[inds_scr].shape)

    print(loss_fn_scr(output[inds_scr], target_scr[inds_scr]))
    print(inds_scr)


    # loss
    if args.scribble:
        loss = args.stepsize_sim * loss_fn(output[inds_sim], target[inds_sim]) + args.stepsize_scr * loss_fn_scr(output[inds_scr], target_scr[inds_scr]) + args.stepsize_con * (lhpy + lhpz)
    else:
        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)

    print('1-3')
    loss.backward()
    optimizer.step()

    print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

    print('1-4')
    if nLabels <= args.minLabels:
        print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break
    print('1-5')

print('2')
# save output image
if not args.visualize:
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
cv2.imwrite("output.png", im_target_rgb)

# f.close()
