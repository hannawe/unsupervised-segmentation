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
import random
import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING']="1"

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False,
                    help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                    help='number of channels')
# parser.add_argument('--maxIter', metavar='T', default=500, type=int,
parser.add_argument('--maxIter', metavar='T', default=100, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=20, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=5, type=float,
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=1, type=float,
                    help='step size for scribble loss')
args = parser.parse_args()


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
# path = r'C:\Users\admin\Desktop\H.We\ML-local\pytorch-unsupervised-segmentation-tip'
# file = path+r'\test.jpg'
# im = cv2.imread(file,1)
# mask = cv2.imread((path+r'\test_scribble.png'),1) # --> (326, 334, 3)
# mask = cv2.imread((path+r'\test_scribble.png'),-1) # --> (326, 334, 4) # there is additional alpha channel
#######


# ---------------------
# setting
# ---------------------
sys.path.insert(1, r'C:\\Users\\admin\\Desktop\\H.We\\ML-local\\pytorch-unsupervised-segmentation-tip')         # insert at 1, 0 is the script path (or '' in REPL)
savePath = r'\\Janeway.institut2b.physik.rwth-aachen.de\User AG Bluhm\H.We\GPU\DATA\pytorch-unsupervised-segmentation-tip\\'


# ---------------------
# load image
# ---------------------
im = cv2.imread(args.input)         # default color=1 gray=0, BGR
data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
if use_cuda:
    data = data.cuda()
data = Variable(data)

# ---------------------
# load scribble image
# ---------------------
if args.scribble:

    mask_ori = cv2.imread(args.input.replace('.' + args.input.split('.')[-1], '_scribble.png'), -1)     # read unchanged including alpha channel e.g. (326, 334, 4)
    if len(mask_ori.shape) > 2:
        mask_ori = cv2.imread(args.input.replace('.' + args.input.split('.')[-1], '_scribble.png'), 0)
    assert(mask_ori.shape[0] == im.shape[0])
    assert(mask_ori.shape[1] == im.shape[1])

    mask = mask_ori.reshape(-1)     # unravel to 1D
    """
    IMPORTANT! mask should be binary except background, i.e. only allow [0, x, 255]
    Here, x=8 (can be any like 17. this value is referred from scribble example)
    
    mask_inds: no repeating brightness and white background -> only unique elements!
    """

    inds_ft_np = np.where((mask != 255) & (mask != 0))[0]
    mask[inds_ft_np] = 8

    target_scr = torch.from_numpy(mask)
    target_scr = Variable(target_scr)                           # requires_grad = false (default)

    mask_inds = np.unique(mask)                                 # debugger view of array, w and h inverted
    mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
    args.minLabels = len(mask_inds)                             # set minLabels, ex) [0 17] --> min cluster#, fixed

    inds_sim = torch.from_numpy(np.where(mask == 255)[0])       # white background
    inds_scr = torch.from_numpy(np.where(mask != 255)[0])       # scribble: not white (must be binary)

    if use_cuda:
        inds_sim = inds_sim.cuda()
        inds_scr = inds_scr.cuda()
        target_scr = target_scr.cuda()

    # To check unique lable of mask
    target_scr_inds_scr = target_scr[inds_scr]
    target_scr_inds_scr = target_scr_inds_scr.data.cpu().numpy()
    print('unique ', np.unique(target_scr_inds_scr))

# ---------------------
# model settings
# ---------------------
model = MyNet(data.size(1))
if use_cuda:
    model.cuda()
model.train()

# loss functions
loss_fn = torch.nn.CrossEntropyLoss()               # similarity loss definition
loss_fn_scr = torch.nn.CrossEntropyLoss()           # scribble loss definition
loss_hpy = torch.nn.L1Loss(size_average=True)       # continuity loss definition
loss_hpz = torch.nn.L1Loss(size_average=True)       # continuity loss definition

# ---------------------
# set target for continuity
# ---------------------
HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], args.nChannel)
HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, args.nChannel)
if use_cuda:
    HPy_target = HPy_target.cuda()
    HPz_target = HPz_target.cuda()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

# lable colours
label_colours = np.random.randint(255, size=(100, 3))

# ---------------------
# unsupervised training
# ---------------------
for batch_idx in range(args.maxIter):

    # forwarding
    optimizer.zero_grad()
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel) # tensor shape: [raveled;  nchannel]
    outputHP = output.reshape((im.shape[0], im.shape[1], args.nChannel)) # tensor shape: [w h ; nchannel]

    """
    contiguous makes a copy of tensor such that the memory layout/order of its elements is the same as if it is created
    a.T will be Fortran contiguous whereas a is C contiguous, and if a.T.reshape(same dim) will cause an error
    since it is not possible to have inconstant stride length (jumps to memory address to access data value)
    need to make a copy and do the operation on that copy
    basically the same operation as: arr2_copy = arr2(=arr.T).copy(), arr2_copy.shape = (12,) --> will not cause error!
    view does not make a copy of the original tensor. It changes the dimensional interpretation (striding) on the original data
    """
    # define loss for continuity
    HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
    HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
    lhpy = loss_hpy(HPy, HPy_target)
    lhpz = loss_hpz(HPz, HPz_target)

    # ---------------------
    # target image: output's max
    # ---------------------
    # similarly bright regions
    ignore, target = torch.max(output, 1)           # max of all elements, 1: row-wise
    im_target = target.data.cpu().numpy()           # no np array

    nLabels = len(np.unique(im_target))

    if args.visualize:

        print("Press 'ESC to stop and do further operations'")

        im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
        cv2.imshow("output", im_target_rgb)

        hi = cv2.waitKey(10) & 0xFF
        if hi==27:
            print("Stopped by user, press 's' for save the current plot, or 'ESC' for exit, 'c' for continue.")

            k = cv2.waitKey(0)
            if k == ord('s'):
                print("Save the current photo to ", savePath, "output.png")
                cv2.imwrite(savePath+"output.png", im_target_rgb)
                cv2.destroyAllWindows()
                break
            if k == 27:  # close on ESC key
                print("Adios")
                cv2.destroyAllWindows()
                break
            if k == ord('s'):  # close on ESC key
                print("Let's goooo")
                continue

    # ---------------------
    # calculate loss
    # ---------------------
    if args.scribble:

        sim_loss = args.stepsize_sim * loss_fn(output[inds_sim], target[inds_sim])
        scr_sim_loss = args.stepsize_scr * loss_fn_scr(output[inds_scr], target_scr[inds_scr])
        conti_loss = args.stepsize_con * (lhpy + lhpz)

        loss = args.stepsize_sim * loss_fn(output[inds_sim], target[inds_sim]) + args.stepsize_scr * loss_fn_scr(output[inds_scr], target_scr[inds_scr]) + args.stepsize_con * (lhpy + lhpz)
        # print("similiarty loss: ", sim_loss)
        # print("scribble simimilarty loss: ", scr_sim_loss)
        # print("continuity loss: ", conti_loss)

    else:
        loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)


    # ---------------------
    # backpropagation
    # ---------------------
    loss.backward()
    optimizer.step()

    print(batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

    if nLabels <= args.minLabels:
        print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break
# ---------------------
# save output image
# ---------------------
if not args.visualize:
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
cv2.imwrite(savePath+"output.png", im_target_rgb)
