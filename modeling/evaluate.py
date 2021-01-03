
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

import video_reader
from model import KickerNet, Variational_L2_loss


def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = KickerNet()
    model.load_state_dict(checkpoint['state_dict'])
    return model

def frame_to_tensor(frame):
    """
    frame: A list of numpy frame of shape ( W,H,C)

    returns: A normalized Tensor of shape (C,W,H)
    """
    frame = cv2.resize(frame, (256, 144))
    frame = frame.T #Rearange to C,W,H
    frame = frame / 255.
    return torch.Tensor(frame)


def normalize_pos(pos):
    m_x, sd_x = 639.5, 369.5040595176188
    m_y, sd_y = 359.5, 207.84589643932512
    return ((pos[0]-m_x )/sd_x,(pos[1]-m_y )/sd_y)

def denormalize_pos(pos):
    m_x, sd_x = 639.5, 369.5040595176188
    m_y, sd_y = 359.5, 207.84589643932512
    return (int(pos[0]*sd_x + m_x),int(pos[1]*sd_y + m_y))

def evaluate(dataset="../dataset/v3.h265", model_name = "trained_models/basicmodel_best.pth.tar", display=True, processed_data=None):
    window_size = 20
    delay = 8
    running = True
    if processed_data:
        image_data = np.load(processed_data)["arr_0"]
        mean = np.mean(image_data, axis=0)
        scale = 128
    else:
        #TODO Better Mean Image
        mean = 0
        scale = 255
    vr = video_reader.VideoReader(dataset)

    model = load_checkpoint(model_name, use_cuda=False)
    model.eval()
    ball_pos = pd.DataFrame(columns=['x', 'y'], dtype=int)
    if display:
        cv2.namedWindow('Frame')

    # Read until video is completed
    while vr.is_opened():
        frame_pos = vr.next_frame
        #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        if running:
            if frame_pos % 50 == 0:
                print(frame_pos)
            # Capture frame-by-frame
            try:
                frame = vr.read_next()
            except StopIteration:
                break

            if processed_data:
                frame_tensor = torch.Tensor(image_data[frame_pos])
            else:
                frame_tensor = frame_to_tensor(frame)
            frame_tensor = torch.unsqueeze((frame_tensor - mean) / scale, 0).float()
            ball_visible, pos = model(frame_tensor)
            denorm_pos = denormalize_pos(pos.detach().cpu().numpy()[0])

            print("ball_visible : {}, Pos: {},  Denorm_pos: {}".format(ball_visible, pos,denorm_pos))

            ball_pos.loc[frame_pos] = {"x": denorm_pos[0], "y": denorm_pos[1]}



        if display:
            point = denorm_pos
            for i in range(window_size):
                past = ball_pos.loc[max(frame_pos - window_size + i, 0)]
                past_point = (int(past['x']), int(past['y']))
                cv2.circle(frame, past_point,  3, (0, 255, 0), 3)
            cv2.circle(frame, point, 5, (0, 0, 255), 8)

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            key = cv2.waitKey(int(delay)) & 0xFF
            # print(key)

        if key == ord('q'):
            break
        elif key == ord('j'):  # slower
            delay *= 1.5
        elif key == ord('k'):  # faster
            delay /= 1.5
        elif key == ord(" "):  # pause
            running = not running
    # When everything done, release the video capture object
    vr.cap.release()
    ball_pos.to_csv(model_name + "evaluated_pos.csv")
    if display:
        # Closes all the frames
        cv2.destroyAllWindows()



if __name__ == '__main__':
    evaluate(processed_data="../dataset/v3rgbgray.npz")
