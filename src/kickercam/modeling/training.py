'''
Recently I have figured out a good training setting:

number of epochs: 150
learning rate schedule: cosine learning rate, initial lr=0.05
weight decay: 4e-5
remove dropout
'''

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.optim as optim

from .config import config
from .data_loader import DataLoader
from .model import KickerNet, Variational_L2_loss

NUM_EPOCHS = 200
BATCH_SIZE = 20
# we first train on v2 and predict/test on v3


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar', name = "basic"):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, name + filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, name + filename),
                        os.path.join(folder, name + 'model_best.pth.tar'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image_data')
    ap.add_argument('labels')
    ap.add_argument('--cuda', action='store_true', default=False)
    args = ap.parse_args(sys.argv[1:])

    NLL_loss = Variational_L2_loss()
    MSE_loss = torch.nn.MSELoss(reduction='none')
    BCE_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.05]))


    net = KickerNet(config)
    if args.cuda:
        net.cuda()
        NLL_loss.cuda()
        MSE_loss.cuda()
        BCE_loss.cuda()
    dl = DataLoader(args.image_data, args.labels)

    # create your optimizer
    #optimizer = optim.SGD(net.parameters(), lr=0.001)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    losses = []
    for epoch in range(NUM_EPOCHS):
        for iteration, (mini_batch, label) in enumerate(dl.iterate_epoch()):
            visible = torch.unsqueeze((label[:,0]>-90).type(dtype=torch.float32), 1)
            if args.cuda:
                label = label.cuda()
                mini_batch = mini_batch.cuda()
                visible = visible.cuda()

            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            #ball_visible, mean_pos, var_pos = net.forward(mini_batch)
            #pos_loss = torch.mean(NLL_loss(mean_pos, label, var_pos) * visible)
            ball_visible, mean_pos= net.forward(mini_batch)
            pos_loss = torch.mean(MSE_loss(mean_pos, label) * visible)


            #print(torch.mean(ball_visible), torch.mean(visible))
            visible_loss = BCE_loss(ball_visible, visible)

            loss = pos_loss + 20*visible_loss
            loss.backward()
            optimizer.step()    # Does the update
            losses.append(( 20*visible_loss.detach().cpu().numpy(), pos_loss.detach().cpu().numpy()))
            print('Train Epoch: {}:{} BCELOSS : {:.6f} \t POSLoss: {:.3f}'.format(
                epoch, iteration, 20 * visible_loss, pos_loss))

        best = max([x[1] for x in losses])
        recent_best = max([x[1] for x in losses[-20:]])
        save_checkpoint({
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, recent_best==best, folder='./trained_models')
    np.savetxt("traininglosses.txt", np.asarray(losses))

if __name__ == '__main__':
    main()
