import cv2
import numpy as np
import torch


def process_frame(frame, use_rgb=True, use_gray=False, target_width=256,
                  target_height=144):
    '''
    Convert a full-size frame (usually 1280x720) into a smaller frame
    (potentially with reduced dimensions).
    '''

    scaled_frame = cv2.resize(frame, (target_width, target_height))
    if use_gray:
        hsv = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2HSV)
        lower_b = np.array([0, 0, 50])
        upper_b = np.array([255, 100, 255])
        # Threshold the HSV image to get only blue colors
        # mask = cv2.inRange(hsv, lower_b, upper_b)
        # Bitwise-AND mask and original image
        # masked_frame = cv2.bitwise_and(scaled_frame, scaled_frame, mask=mask)
        gray = np.expand_dims(cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY).T,
                              0)

    #TODO Is it good to Transpose all images? W,H,C -> C,H,W, but we want C,W,H ?! 
    if use_rgb and use_gray:
        result = np.concatenate([scaled_frame.T, gray], axis=0)
    elif use_gray:
        result = gray
    else:
        result = scaled_frame.T
    return result

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
    if pos[0] < 0:
        return (-100,-100)
    m_x, sd_x = 639.5, 369.5040595176188
    m_y, sd_y = 359.5, 207.84589643932512
    return ((pos[0]-m_x )/sd_x,(pos[1]-m_y )/sd_y)

def normalize_image(image, mean, scale=128.):
    return torch.Tensor((image-mean) / scale)