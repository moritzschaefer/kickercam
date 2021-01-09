'''
This is completely copied from https://github.com/tonylins/pytorch-mobilenet-v2

it needs to be adapted entirely
'''

import numpy as np
import pandas as pd
import torch

from ..preprocessing import normalize_pos, normalize_image

EPOCHSIZE = 20


class DataLoader:
    def __init__(self, dataset, label_file):
        '''
        load an npz (for images) and a csv (for labels) file for training (and prediction)
        '''
        self.dataset = dataset
        self.image_data = np.load(dataset)["arr_0"]  # arr_0 is the default name
        self.mean = np.mean(self.image_data,axis=0)
        self.scale = 128
        self.iteration = 0
        self.df = pd.read_csv(label_file)
        self.epoch_size = EPOCHSIZE

    def iterate_epoch(self, batch_size=40):
        for _ in range(self.epoch_size):   #  self.iteration % EPOCHSIZE >0:
            pos = np.random.randint(0, len(self.df), batch_size)
            labels = torch.Tensor([normalize_pos((self.df.loc[frame_pos]['x'],
                                                  self.df.loc[frame_pos]['y']))
                                   for frame_pos in pos])

            images = normalize_image(self.image_data[pos], self.mean, self.scale) # TODO use frame_to_tensor and/or integrate this normalization into preprocessing/__init__.py
            self.iteration += 1
            yield images, labels
