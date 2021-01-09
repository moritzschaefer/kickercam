import cv2
import numpy as np


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

    if use_rgb and use_gray:
        result = np.concatenate([scaled_frame.T, gray], axis=0)
    elif use_gray:
        result = gray
    else:
        result = scaled_frame.T
    return result
