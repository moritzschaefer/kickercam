import cv2
import numpy as np
import pandas as pd


def load_images(video_fn, label_fn, target_width, target_height, normalize_values=True):
    y = pd.read_csv(label_fn).to_numpy()  # TODO need to normalize still!!!
    X = np.ndarray((y.shape[0], target_width,target_height, 3))

    cap = cv2.VideoCapture(video_fn)
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    for frame_i in range(X.shape[0]):
        ret, frame = cap.read()
        if ret is not True:
            raise RuntimeError('no more frames in video -.-')

        frame = cv2.resize(frame, (target_width,target_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = np.asarray(frame)
        frame = frame.astype("float")
        if normalize_values:
            frame /= 255.0

        X[frame_i, :, :, :] = np.transpose(frame, axes=[1,0,2])

    return X, y
