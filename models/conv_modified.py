import torch.nn as nn
import torch
import numpy as np


class Conv3x3_mofied(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1, dilation=1,use_expansion=False):
        super(Conv3x3_mofied, self).__init__()
        # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.conv2d_3x3 = conv3x3(in_planes, out_planes, stride=stride, groups=groups, dilation=dilation)
        self.expansion_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,bias=False)
        # nn.init.constant_(self.expansion_1x1.weight.data, 0.0)
        # self.expansion_1x1.weight.data.zero_()

        self.use_expansion = use_expansion
    def set_expansion(self, use_expansion=True):
        self.use_expansion = use_expansion

    def re_param(self):
        kernel = self.get_equivalent_kernel_bias()
        self.conv2d_3x3.weight.data = kernel
        # self.expansion_1x1.weight.data.zero_()
        self.zero_expansion()

    def zero_expansion(self):
        # pass
        nn.init.constant_( self.expansion_1x1.weight.data,0.0)
        # self.expansion_1x1.weight.data.zero_()

    def forward(self, x):
        if not self.use_expansion:
            return self.conv2d_3x3(x)
        else:
            with torch.no_grad():
                out1 = self.conv2d_3x3(x)
            return self.expansion_1x1(x) + out1



    def get_equivalent_kernel_bias(self):
        # bias no use
        kernel3x3, _ = self._fuse_bn_tensor(self.conv2d_3x3)
        kernel1x1, _ = self._fuse_bn_tensor(self.expansion_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Module):
            kernel = branch.weight
            return kernel, kernel
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


