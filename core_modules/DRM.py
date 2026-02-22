import torch
import torch.nn as nn
import torch.nn.functional as F

class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0*g:1*g, 1, 2] = 1.0
        mask[:, 1*g:2*g, 1, 0] = 1.0
        mask[:, 2*g:3*g, 2, 1] = 1.0
        mask[:, 3*g:4*g, 0, 1] = 1.0
        mask[:, 4*g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1) 
        return y

class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0*g:1*g, 0, 1, 2] = 1.0 ## left
        self.weight[1*g:2*g, 0, 1, 0] = 1.0 ## right
        self.weight[2*g:3*g, 0, 2, 1] = 1.0 ## up
        self.weight[3*g:4*g, 0, 0, 1] = 1.0 ## down
        self.weight[4*g:, 0, 1, 1] = 1.0 ## identity     

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y) 
        return y

class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory': 
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y


class DRblock(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super().__init__()
        self.SConv1 = ShiftConv2d(inp_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.SConv2 = ShiftConv2d(out_channels, out_channels)

    def forward(self, x):
        x = self.SConv1(x)
        x = self.relu(x)
        x = self.SConv2(x)
        return x

class DRM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.branch1 = nn.Conv2d(c1, c2, kernel_size=1)

        self.branch2_conv1x1 = nn.Conv2d(c1, c2, kernel_size=1)
        self.dr_blocks = nn.Sequential(
            DRblock(c2, c2),
            DRblock(c2, c2),
            DRblock(c2, c2)
        )
        self.branch2_conv1x1_out = nn.Conv2d(c2, c2, kernel_size=1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2_conv1x1(x)
        x2 = self.dr_blocks(x2)
        x2 = self.branch2_conv1x1_out(x2)
        out = x1 + x2
        return out
