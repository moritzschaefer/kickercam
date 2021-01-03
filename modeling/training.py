'''
Recently I have figured out a good training setting:

number of epochs: 150
learning rate schedule: cosine learning rate, initial lr=0.05
weight decay: 4e-5
remove dropout
'''

import torch.optim as optim
import torch
import os
import shutil
from model import KickerNet, Variational_L2_loss
from data_loader import DataLoader
import numpy as np
NUM_EPOCHS = 200
BATCH_SIZE = 20
# we first train on v2 and predict/test on v3

def main():
    net = KickerNet()

    #dl = DataLoader("../dataset/v2.h265", "../dataset/v2.h265.csv")
    dl = DataLoader("../dataset/v2grey.npz", "../dataset/v2.h265.csv")

    # create your optimizer
    #optimizer = optim.SGD(net.parameters(), lr=0.001)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    losses = []
    NLL_loss = Variational_L2_loss()
    MSE_loss = torch.nn.MSELoss(reduction='none')
    BCE_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.05]))
    for epoch in range(NUM_EPOCHS):
        iteration = 0
        while dl.running_epoch:
            mini_batch, label = dl.get_batch(BATCH_SIZE)

            visible = torch.unsqueeze((label[:,0]>-90).type(dtype=torch.float32), 1)
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
            iteration += 1
        dl.running_epoch = True

        save_checkpoint({
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, True, folder='./trained_models')
    np.savetxt("traininglosses.txt",np.asarray(losses))

def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar', name = "basic"):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, name + filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, name + filename),
                        os.path.join(folder, name + 'model_best.pth.tar'))


if __name__ == '__main__':
    main()