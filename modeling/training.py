'''
Recently I have figured out a good training setting:

number of epochs: 150
learning rate schedule: cosine learning rate, initial lr=0.05
weight decay: 4e-5
remove dropout
'''

from torch.optim import optim

from data_loader import DataLoader
from model import KickerNet

NUM_EPOCHS = 100

# we first train on v2 and predict/test on v3

def class_regression_loss(prediction, y)

def main():
    net = KickerNet()
    dl = DataLoader(fn)

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(NUM_EPOCHS):
        for mini_batch, y in dl:  # dl needs to return an iterator

            # TODO compute a loss dependent on whether the ball is in the field or not

            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            output = net.forward(mini_batch)
            loss = class_regression_loss(output, y)
            loss.backward()
            optimizer.step()    # Does the update
