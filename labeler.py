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
    elif event == cv2.EVENT_MBUTTONDOWN:
        delay *= 1.7
    elif event == cv2.EVENT_RBUTTONDOWN:
        drop_last = True



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
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, frame = cap.read()
    cv2.imshow('Frame', frame)
    cv2.waitKey(1600)  # initial wait to position mouse
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read until video is completed
    while(cap.isOpened()):
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if running:
            if frame_pos % 1 == 0:
                print(frame_pos)

            # Capture frame-by-frame
            ret, frame = cap.read()
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
            ball_pos =ball_pos.append(pd.Series(index=frame_pos,data={"x":-1, "y": -1})).sort_index()

        elif key == ord('j'):
            delay *= 1.5
        elif key == ord('k'):  # faster
            delay /= 1.5
        elif key == ord('h') or key == ord('H'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0,frame_pos - 100 * 10**(key == ord('B'))))
            ret, frame = cap.read()
            if ret is not True:
                break
            cv2.imshow('Frame', frame)
            cv2.waitKey(100)
        elif key == ord(" "):   # pause
            running = not running
        if track_point:

            ball_pos.loc[frame_pos] = [pos[0],pos[1]]
            ball_pos = ball_pos.sort_index()
            #ball_pos = ball_pos.append(pd.Series(index=frame_pos, data={"x": pos[0], "y": pos[1]})).sort_index()
            track_point = False

        if drop_last:
            ball_pos.drop(index=ball_pos.index[-1], inplace=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, ball_pos.index[-1]))
            print("Droped the last Frame with index",  int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

            ret, frame = cap.read()
            if ret is not True:
                break
            cv2.imshow('Frame', frame)
            cv2.waitKey(100)

            print("After Getting Image index:",  int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
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

    args = ap.parse_args(sys.argv[1:])

    label_file = args.label_file
    if not label_file:
        label_file = args.input_video + '_labels.csv'
    try:
        # TODO
        pre_ball_pos = pd.read_csv(label_file, index_col=0).sort_index()
        #pre_ball_pos = pd.read_csv(label_file).apply(lambda row: (row['x'], row['y']),
        #                                            axis=1).todict()
    except FileNotFoundError:
        pre_ball_pos = pd.DataFrame()

    if not args.start_frame:
        pass
    ball_pos = label(args.input_video, args.start_frame, pre_ball_pos)

    ball_pos.to_csv(label_file)

    # pd.DataFrame({'x': [v[0]  if v else None for v in ball_pos], 'y': [v[1]  if v else None for v in ball_pos]}).to_csv(label_file)
