import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import sys
import shutil

lr = 0.0001
batch_size = 300

class Differential_AE(nn.Module):
    def __init__(self, input_dim, hidden_size, n_layer):
        super(Differential_AE, self).__init__()
        net = []
        cur_dim = input_dim[0]
        for i in range(n_layer):
            net.append(nn.Conv2d(cur_dim, hidden_size*(i+1), 3, 1, 1))
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


def load_data(data_set = None):
    return torch.Tensor(np.zeros(10,1,28,28))



def load_and_train():
    if not os.path.isdir('./trained_models'):
        os.makedirs('./trained_models')
    train_data = load_data()
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True)
    cuda = torch.cuda.is_available()
    model = Differential_AE(32, 128,3)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ae_loss = nn.L1Loss()
    reg_loss = lambda x: torch.norm(x)
    pos_loss = nn.MSELoss()

    if cuda:
        model.cuda()
        ae_loss.cuda()
        pos_loss.cuda()

    def train(epoch):
        model.train()

        # NOTE: is_paired is 1 if the example is paired
        for batch_idx, (image, pos) in enumerate(train_loader):
            if cuda:
                image = image.cuda()
                pos = pos.cuda()
            image = Variable(image)
            pos = Variable(pos)
            # refresh the optimizer
            optimizer.zero_grad()

            # pass data through model
            recon_image, pred_pos, low_mu = model(image, pos)



            # compute ELBO for each data combo
            joint_loss = elbo_loss(recon_image_1, image, recon_text_1, text, mu_1, logvar_1,
                                   lambda_image=lambda_image, lambda_text=lambda_text,
                                   annealing_factor=annealing_factor, prob_image=probabilistic_out)
            image_loss = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2,
                                   lambda_image=lambda_image, lambda_text=lambda_text,
                                   annealing_factor=annealing_factor, prob_image=probabilistic_out)
            text_loss = elbo_loss(None, None, recon_text_3, text, mu_3, logvar_3,
                                  lambda_image=lambda_image, lambda_text=lambda_text,
                                  annealing_factor=annealing_factor)
            train_loss = joint_loss + image_loss + text_loss
            train_loss_meter.update(train_loss, batch_size)

            if torch.isnan(train_loss).any():
                raise Exception(
                    "There is an Nan Loss:" + str(torch.mean(recon_image_1._natural_params[0])) + str(torch.mean(recon_image_1._natural_params[1])) +
                          str(torch.mean(recon_image_2._natural_params[0])) +str(torch.mean(recon_image_2._natural_params[1])) +str(torch.min(train_loss)))

            # compute gradients and take step
            train_loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:

                if probabilistic_out:
                    print(torch.mean(recon_image_1._natural_params[0]), torch.mean(recon_image_1._natural_params[1]),
                          torch.mean(recon_image_2._natural_params[0]), torch.mean(recon_image_2._natural_params[1]))

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAnnealing-Factor: {:.3f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), train_loss_meter.avg, annealing_factor))
                _run.log_scalar("train_loss",  float(train_loss_meter.avg.detach().cpu()), epoch * len(train_loader.dataset) + batch_idx * len(image))
                if probabilistic_out:
                    _run.log_scalar("alpha", float(torch.mean(recon_image_1._natural_params[0]).detach().cpu()),
                                epoch * len(train_loader.dataset) + batch_idx * len(image))
                    _run.log_scalar("beta", float(torch.mean(recon_image_1._natural_params[1]).detach().cpu()),
                                    epoch * len(train_loader.dataset) + batch_idx * len(image))
        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, train_loss_meter.avg))

    def test(epoch):
        model.eval()
        test_loss_meter = AverageMeter()

        for batch_idx, (image, text) in enumerate(test_loader):
            if cuda:
                image = image.cuda()
                text = text.cuda()

            if probabilistic_out:
                image = image*0.99 + 0.005
                assert ((image > 0).all() and (image < 1).all())
            image = Variable(image)
            text = Variable(text)
            batch_size = len(image)

            recon_image_1, recon_text_1, mu_1, logvar_1 = model(image, text)
            recon_image_2, recon_text_2, mu_2, logvar_2 = model(image)
            recon_image_3, recon_text_3, mu_3, logvar_3 = model(text=text)
            joint_loss = elbo_loss(recon_image_1, image, recon_text_1, text, mu_1, logvar_1, prob_image=probabilistic_out)
            image_loss = elbo_loss(recon_image_2, image, None, None, mu_2, logvar_2, prob_image=probabilistic_out)
            text_loss = elbo_loss(None, None, recon_text_3, text, mu_3, logvar_3, prob_image=probabilistic_out)
            test_loss = joint_loss + image_loss + text_loss
            test_loss_meter.update(test_loss.data, batch_size)

        print('====> Test Loss: {:.4f}'.format(test_loss_meter.avg))
        _run.log_scalar("Test_loss", float(test_loss_meter.avg.detach().cpu().numpy()), epoch * len(train_loader.dataset))
        return test_loss_meter.avg

    best_loss = sys.maxsize
    for epoch in range(1, epochs + 1):
        train(epoch)
        test_loss = test(epoch)
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        # save the best model and current model
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'n_latents': n_latents,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder='./trained_models')


    pass


