import torch
import numpy as np
import torch.nn as nn



class Differential_AE(nn.Module):
    def __init__(self, input_dim, hidden_size, n_layer):
        super(Differential_AE, self).__init__()
        net = []
        cur_dim = input_dim[0]
        for i in range(n_layer):
            net.append(nn.Conv2d(cur_dim, hidden_size*(i+1),3,1,1))
            net.append(nn.LeakyReLU(0.1))
            net.append(nn.BatchNorm2d(cur_dim))
            cur_dim = hidden_size * (i + 1)
            net.append(nn.Dropout2d(0.2))
            net.append(nn.MaxPool2d(2))

        net.append(nn.Conv2d(cur_dim, hidden_size,3,1,1))
        net.append(nn.LeakyReLU(0.1))
        net.append(nn.Conv2d(hidden_size, hidden_size, 1, 1, 0))

        self.down_net = nn.Sequential(*net)

        up_net = []
        up_net.append(nn.Conv2d(hidden_size, cur_dim, 1, 1, 0))
        up_net.append(nn.LeakyReLU(0.1))
        for i in range(n_layer):
            up_net.append(nn.Conv2d(cur_dim,hidden_size*(n_layer-i),3,1,1))
            cur_dim = hidden_size*(n_layer-i)
            net.append(nn.LeakyReLU(0.1))
            net.append(nn.Dropout2d(0.2))
            net.append(nn.UpsamplingNearest2d(2))
        up_net.append(nn.Conv2d(cur_dim,input_dim,1,1,0))
        self.up_net = nn.Sequential(*up_net)

        pos_net = []
        cur_dim = hidden_size
        for i in range(n_layer):
            pos_net.append(nn.Conv2d(cur_dim,cur_dim*2, 3, 1, 1))
            cur_dim = cur_dim*2
            pos_net.append(nn.LeakyReLU(0.1))
            pos_net.append(nn.Dropout2d(0.2))
            pos_net.append(nn.MaxPool2d(2))

        pos_net.append(nn.Flatten())
        rest_shape = input_dim[1] // 2**(2*n_layer)
        pos_net.append(nn.Linear(rest_shape**2*cur_dim,cur_dim))
        pos_net.append(nn.Linear(cur_dim,6))
        self.pos_net = nn.Sequential(*pos_net)

    def forward(self, x):
        low = self.down_net(x)
        recons = self.up_net(low)
        pos = self.pos_net(low)
        return recons, pos, low


def train(x,y):
    pass


