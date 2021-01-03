'''
The model processes a 128x64 image.
This corresponds to approximately a quarter of a 5x downscaled original image (1280x720)
It outputs 3 values:
- x-position of the ball
- y-position of the ball
- classification if the ball is visible in the image or not
Adopted from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
'''

import math
import torch
import torch.nn as nn
import numpy as np

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    # TODO replace batchnorm with instance norm?
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class KickerNet(nn.Module):
    def __init__(self, input_size=(256, 144), width_mult=1.):
        super(KickerNet, self).__init__()
        block = InvertedResidual
        input_channel = 16
        last_channel = 16
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 32, 4, 2],
            [6, 16, 4, 2],
        ]
        downscaling_factor = np.prod([s for _, _, _, s in interverted_residual_setting])
        self.last_channel = (input_size[0] * input_size[1]) // downscaling_factor**2 * interverted_residual_setting[-1][1]
        # building first layer
        assert input_size[0] % downscaling_factor == 0
        assert input_size[1] % downscaling_factor == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.features = [conv_bn(4, input_channel, 1)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        #self.features.append(nn.Conv2d(input_channel, input_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        #self.classifier = nn.Linear(self.last_channel, 2)  # predict whether a ball is visible or not
        self.classifier = nn.Linear(input_channel, 1)  # predict whether a ball is visible or not
        self.regressor = nn.Sequential(nn.Linear(self.last_channel, 64), nn.Linear(64, 2)) # predict the (x, y) ball position

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        pooled_x = torch.max(torch.max(x, 3).values, 2).values
        x = x.view(-1, self.last_channel)
        ball_visible = self.classifier(pooled_x)

        ball_pos = self.regressor(x)
        #mean_pos, var_pos = torch.chunk(ball_pos, chunks=2, dim=1)
        #var_pos = nn.functional.softplus(var_pos)
        return ball_visible, ball_pos  # mean_pos, var_pos

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



class Variational_L2_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loc, target, scale=None):
        if scale is None:
            loc, scale = torch.chunk(loc, 2, 1)
        # print("Variational Input", torch.mean(loc), torch.mean(scale))
        # print("Variational loss", torch.mean((loc-target)**2 / (scale + 10**-4) + torch.log(scale + 10**-4)))
        return torch.mean((loc - target) ** 2 / (scale + 10 ** -4) + torch.log((scale + 10 ** -4)), dim=1)# ** 2))



if __name__ == '__main__':
    net = KickerNet(width_mult=1)
