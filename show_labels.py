import math

import cv2
import pandas as pd
import argparse
import sys
import collections

RESIZE = 1  # set to 1 to disable scaling
SHOW_BALL_LABEL = True

# 6800 -> 7250
def show_labels(video_fn, label_fn, window_size = 30):

    delay = 8
    cap = cv2.VideoCapture(video_fn)

    # Check if camera opened successfully
    if cap.isOpened() == False: 
        print("Error opening video stream or file")


    cv2.namedWindow('Frame') 
    print(label_fn)
    df = pd.read_csv(label_fn)
    df.fillna(-1, inplace=True)
    df['x'] = df['x'].astype(int)
    df['y'] = df['y'].astype(int)

    ball_pos = []
    # Read until video is completed
    frame_pos = -1
    while cap.isOpened() and frame_pos + 2 < len(df):
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_pos % 50 == 0:
            print(frame_pos)
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is not True:
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
        elif key == ord('b'):  # back
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(1,frame_pos-10))
        elif key == ord('B'):  # back 100
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, frame_pos - 100))
    # When everything done, release the video capture object
    cap.release()

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
