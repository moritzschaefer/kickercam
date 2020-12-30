#!/usr/bin/env python3
import math

import cv2
import pandas as pd
import argparse
import sys
import collections

from video_reader import VideoReader


RESIZE = 1  # set to 1 to disable scaling
SHOW_BALL_LABEL = True
running = True
# 6800 -> 7250
def show_labels(video_fn, label_fn, window_size = 30):
    global running
    delay = 8
    vr = VideoReader(video_fn)
    cv2.namedWindow('Frame') 
    print(label_fn)
    df = pd.read_csv(label_fn)
    df.fillna(-1, inplace=True)
    df['x'] = df['x'].astype(int)
    df['y'] = df['y'].astype(int)

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

        if RESIZE and RESIZE != 1:
            frame = cv2.resize(frame, None, fx=RESIZE, fy=RESIZE)


        pos = df.loc[frame_pos]
        if SHOW_BALL_LABEL and pos['x'] >= 0:
            point = (int(pos['x'] * RESIZE), int(pos['y'] * RESIZE))
            for i in range(window_size):
                past = df.loc[max(frame_pos - window_size + i, 0)]
                past_point = (int(past['x'] * RESIZE), int(past['y'] * RESIZE))
                cv2.circle(frame, past_point,  3, (0, 0, 255), 3)
            past_point = (int(df.loc[max(frame_pos-window_size,0)]['x'] * RESIZE), int(df.loc[max(frame_pos-window_size,0)]['y'] * RESIZE))
            cv2.line(frame, point, past_point, (0, 255, 0), math.ceil(4*RESIZE))
            cv2.circle(frame, point, math.ceil(5*RESIZE), (0, 0, 255), math.ceil(8*RESIZE))

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
        elif key == ord('h'):
            vr.jump_back(50)
            try:
                frame = vr.read_next()
            except StopIteration:
                break
            cv2.imshow('Frame', frame)
            cv2.waitKey(100)
        elif key == ord('g'):
            vr.jump_back(150)
            try:
                frame = vr.read_next()
            except StopIteration:
                break
            cv2.imshow('Frame', frame)
            cv2.waitKey(100)
        elif key == ord(" "):  # pause
            running = not running
    # When everything done, release the video capture object
    vr.cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input_video', metavar='input-video', type=str)
    ap.add_argument('--label-file', type=str, default=None)
    ap.add_argument('--start-frame', type=int, default=0)

    args = ap.parse_args(sys.argv[1:])

    label_file = args.label_file
    if not label_file:
        label_file = args.input_video + '_labels.csv'
    try:
        pre_ball_pos = pd.read_csv(label_file).apply(lambda row: (row['x'], row['y']),
                                                     axis=1).tolist()
    except FileNotFoundError:
        print("labels not found")
        pre_ball_pos = []
    ball_pos = show_labels(args.input_video, label_file)
