'''
This is completely copied from https://github.com/tonylins/pytorch-mobilenet-v2

it needs to be adapted entirely
'''

import numpy as np
import pandas as pd
import torch

EPOCHSIZE = 20

def frame_to_tensor(frame):
    """
    frame: A list of numpy frame of shape ( W,H,C)

    returns: A normalized Tensor of shape (C,W,H)
    """
    import cv2
    frame = cv2.resize(frame, (256, 144))

    frame = frame.T #Rearange to C,W,H
    #frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.
    tensor_frame = torch.Tensor(frame)

    return(tensor_frame)




def normalize_pos(pos):
    if pos[0] < 0:
        return (-100,-100)
    m_x, sd_x = 639.5, 369.5040595176188
    m_y, sd_y = 359.5, 207.84589643932512
    return ((pos[0]-m_x )/sd_x,(pos[1]-m_y )/sd_y)

class DataLoader:
    def __init__(self, dataset, label_file):
        self.dataset = dataset
        if ".npz" in dataset:
            self.load_from_video = False
            self.image_data = np.load(dataset)["arr_0"]
            self.mean = np.mean(self.image_data,axis=0)
            self.iteration = 0
        else:
            self.load_from_video = True
            import video_reader
            self.vr = video_reader.VideoReader(dataset)
        self.df = pd.read_csv(label_file)
        self.running_epoch = True

    def get_batch(self, batch_size = 40):
        self.running_epoch = True
        if self.load_from_video:
            images = torch.zeros(batch_size, 3, 256, 144)
            labels = torch.zeros(batch_size, 2)
            for i in range(batch_size):
                try:
                    frame_pos = self.vr.next_frame
                    if frame_pos >= len(self.df): #More frames than we have labels
                        self.running_epoch = False
                        self.vr = video_reader.VideoReader(self.dataset)
                        frame_pos = self.vr.next_frame
                    image = self.vr.read_next()
                    images[i] = frame_to_tensor(image)
                    labels[i] = torch.Tensor([normalize_pos((self.df.loc[frame_pos]['x'], self.df.loc[frame_pos]['y']))])

                    #skip 1-500 frames:
                    for i in range(np.random.randint(1,500)):
                        _ = self.vr.read_next()
                except StopIteration:
                    self.running_epoch = False
                    self.vr = video_reader.VideoReader(self.dataset)
        else:
            pos = np.random.randint(0,len(self.df),batch_size)
            labels = torch.Tensor([normalize_pos((self.df.loc[frame_pos]['x'], self.df.loc[frame_pos]['y'])) for frame_pos in pos])
            images = torch.Tensor((self.image_data[pos]-self.mean) / 128.)
            self.iteration += 1
            if self.iteration % EPOCHSIZE == 0:
                self.running_epoch = False


        return images, labels
