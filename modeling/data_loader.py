'''
This is completely copied from https://github.com/tonylins/pytorch-mobilenet-v2

it needs to be adapted entirely
'''

import numpy as np
import pandas as pd
import torch

from ..preprocessing import frame_to_tensor

EPOCHSIZE = 20



def normalize_pos(pos):
    if pos[0] < 0:
        return (-100,-100)
    m_x, sd_x = 639.5, 369.5040595176188
    m_y, sd_y = 359.5, 207.84589643932512
    return ((pos[0]-m_x )/sd_x,(pos[1]-m_y )/sd_y)

class DataLoader:
    def __init__(self, dataset, label_file):
        '''
        load an npz (for images) and a csv (for labels) file for training (and prediction)
        '''
        self.dataset = dataset
        self.image_data = np.load(dataset)["arr_0"]  # arr_0 is the default name
        self.mean = np.mean(self.image_data,axis=0)
        self.iteration = 0
        self.df = pd.read_csv(label_file)


    def iterate_epoch(self, batch_size=40):
        while True:
            pos = np.random.randint(0, len(self.df), batch_size)
            labels = torch.Tensor([normalize_pos((self.df.loc[frame_pos]['x'],
                                                  self.df.loc[frame_pos]['y']))
                                   for frame_pos in pos])

            images = torch.Tensor((self.image_data[pos]-self.mean) / 128.)  # TODO use frame_to_tensor and/or integrate this normalization into preprocessing/__init__.py
            self.iteration += 1
            if self.iteration % EPOCHSIZE == 0:
                raise StopIteration

            return images, labels
