#!/usr/bin/env python3
import argparse
import sys
import time

import cv2
import pandas as pd

# label the video by mouse dragging

pos = None
drop_last = None
delay = 8
track_point = False
running = True

def mousePosition(event, x, y, flags, passed_param):
    global pos, drop_last, delay, track_point
    if event == cv2.EVENT_MOUSEMOVE:
        # print(x,y)
        pos = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        track_point = True
    elif  event == cv2.EVENT_LBUTTONUP:
        track_point = False
    elif event == cv2.EVENT_MBUTTONDOWN:
        delay *= 1.7
    elif event == cv2.EVENT_RBUTTONDOWN:
        drop_last = True

def read(cap):
    # 0-based index of the frame to be decoded/captured next.
    frame_pos = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    ret, frame = cap.read()

    return ret, frame, frame_pos



def label(filename, start_frame=0, ball_pos=None):

    global pos, drop_last, delay, track_point, running
    ball_pos = ball_pos.sort_index()
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(filename)

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    cv2.namedWindow('Frame')

    cv2.setMouseCallback('Frame', mousePosition)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_frame)

    ret, frame, frame_pos = read(cap)
    cv2.imshow('Frame', frame)
    cv2.waitKey(1600)  # initial wait to position mouse
    #cap.set(cv2.CAP_PROP_POS_MSEC, start_frame)
    # Read until video is completed
    while(cap.isOpened()):

        if running:
            if frame_pos % 50 == 0:
                print(frame_pos)

            # Capture frame-by-frame
            ret, frame, frame_pos = read(cap)
            if ret is not True:
                break

        # show last label
        if len(ball_pos) > 0:
            cv2.circle(frame, tuple(ball_pos.iloc[-1].values), 3, (0, 0, 255), 3)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        key = cv2.waitKey(int(delay)) & 0xFF
        # print(key)

        if key == ord('q'):
            break
        elif key == ord('u'):
            ball_pos.loc[frame_pos] ={"x":-1, "y": -1}  # allows overwrite
            ball_pos.sort_index(inplace=True)
            ball_pos.drop(ball_pos.index[ball_pos.index > frame_pos], inplace=True)

        elif key == ord('j'):
            delay *= 1.5
        elif key == ord('k'):  # faster
            delay /= 1.5
        elif key == ord('h') or key == ord('g'):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0,frame_pos - 500 - 1000 *(key == ord('g'))))
            ret, frame, frame_pos = read(cap)
            if ret is not True:
                break
            cv2.imshow('Frame', frame)
            cv2.waitKey(100)
        elif key == ord(" "):   # pause
            running = not running
        if track_point:
            #ball_pos.loc[frame_pos] = [pos[0], pos[1]]
            #ball_pos = ball_pos.sort_index()
            ball_pos.loc[frame_pos] = {"x": pos[0], "y": pos[1]}  # allows overwrite
            ball_pos.sort_index(inplace=True)
            ball_pos.drop(ball_pos.index[ball_pos.index > frame_pos], inplace=True)

        if drop_last and len(ball_pos) > 0:
            print(f"Last index before drop {ball_pos.index[-1]}")
            ball_pos = ball_pos.drop(index=ball_pos.index[-1])
            frame_pos = ball_pos.index[-1]
            print(f"Last index After drop {ball_pos.index[-1]}")

            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, frame_pos)
            except IndexError:
                cap.set(cv2.CAP_PROP_POS_MSEC, 0)
            #print("Dropped the last Frame with index",  int(cap.get(cv2.CAP_PROP_POS_MSEC)))

            ret, frame, frame_pos = read(cap)
            if ret is not True:
                break
            cv2.imshow('Frame', frame)
            cv2.waitKey(100)

            #print("After Getting Image index:",  int(cap.get(cv2.CAP_PROP_POS_MSEC)))
            drop_last = False


    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    return ball_pos

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input_video', metavar='input-video', type=str)
    ap.add_argument('--label-file', type=str, default=None)
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--interpolated-label-file', type=str, default=None)

    args = ap.parse_args(sys.argv[1:])

    label_file = args.label_file
    interpolated_file = args.interpolated_label_file
    if not label_file:
        label_file = args.input_video + '_labels.csv'
    if not interpolated_file:
        interpolated_file = args.input_video + '_interpolated_labels.csv'
    try:
        # TODO
        pre_ball_pos = pd.read_csv(label_file, index_col=0).sort_index()
        #pre_ball_pos = pd.read_csv(label_file).apply(lambda row: (row['x'], row['y']),
        #                                            axis=1).todict()
    except FileNotFoundError:
        pre_ball_pos = pd.DataFrame(columns=['x', 'y'], dtype=int)

    if not args.start_frame:
        pass
    ball_pos = label(args.input_video, args.start_frame, pre_ball_pos)

    ball_pos.to_csv(label_file)

    from interpolate import interpolate
    interpolate(ball_pos).to_csv(interpolated_file)


    # pd.DataFrame({'x': [v[0]  if v else None for v in ball_pos], 'y': [v[1]  if v else None for v in ball_pos]}).to_csv(label_file)
