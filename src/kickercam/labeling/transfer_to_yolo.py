#!/usr/bin/env python3
import math

import cv2
import pandas as pd
import argparse
import sys
import collections
import os
from video_reader import VideoReader
from pathlib import Path

RESIZE = 1  # set to 1 to disable scaling
SHOW_BALL_LABEL = True
running = True
# 6800 -> 7250
def show_labels(video_fn, label_fn, start_pos = -1, window_size = 30):
    global running
    delay = 8
    vr = VideoReader(video_fn)
    print(label_fn)
    df = pd.read_csv(label_fn)
    df.fillna(-1, inplace=True)
    df['x'] = df['x'].astype(int)
    df['y'] = df['y'].astype(int)
    if not os.path.exists("yolo/images/train/"):
        os.makedirs("yolo/images/train/")
        os.makedirs("yolo/labels/train/")

    ball_pos = []
    # Read until video is completed
    frame_pos = -1
    while vr.is_opened() and frame_pos + 2 < len(df):
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
        if frame_pos <= start_pos:
            continue
        
        frame_size = frame.shape
        #Resize_x = 428 / frame_size[0]
        #Resize_y = 640 / frame_size[1]
        frame = cv2.resize(frame, (640, 428))

        pos = df.loc[frame_pos]
        point = (pos['x'] /frame_size[1] , pos['y'] / frame_size[0])

        cv2.imwrite(f"yolo/images/train/{Path(video_fn).stem}{frame_pos}.jpg", frame)
        if point[0] > 0 and point[1] >0:
            f = open(f"yolo/labels/train/{Path(video_fn).stem}{frame_pos}.txt", "w")
            f.write(f"0 {point[0]} {point[1]} {0.032548*2} {0.053971*2}")
            f.close()


    # When everything done, release the video capture object
    vr.cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input_video', metavar='input-video', type=str)
    ap.add_argument('--label-file', type=str, default=None)
    ap.add_argument('--start-frame', type=int, default=-1)

    args = ap.parse_args(sys.argv[1:])

    label_file = args.label_file
    if not label_file:
        label_file = args.input_video + '.csv'
    try:
        pre_ball_pos = pd.read_csv(label_file, index_col=0).apply(lambda row: (row['x'], row['y']),
                                                     axis=1).tolist()
    except FileNotFoundError:
        print("labels not found")
        pre_ball_pos = []
    ball_pos = show_labels(args.input_video, label_file, args.start_frame)
