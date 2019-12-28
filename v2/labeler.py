import sys
import time

import cv2
import pandas as pd

# label the video by mouse dragging

pos = None
go_back = None
delay = 200


def mousePosition(event,x,y,flags, passed_param):
    global pos, go_back, delay
    if event == cv2.EVENT_MOUSEMOVE:
        # print(x,y)
        pos = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        go_back = True
    elif event == cv2.EVENT_MBUTTONDOWN:
        delay *= 1.7


def label(filename, start_frame=0, ball_pos=[]):

    global pos, go_back, delay

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
    cv2.waitKey(1600)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read until video is completed
    while(cap.isOpened()):
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_pos % 50 == 0:
            print(frame_pos)

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is not True:
            break

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        key = cv2.waitKey(int(delay)) & 0xFF
        # print(key)

        if key == ord('q'):
            break
        elif key == ord('u'):
            pos = (-1, -1)
        elif key == ord('j'):
            delay *= 1.5
        elif key == ord('d'):  # faster
            delay /= 1.5
        elif key == ord('h') or go_back:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos - 40)
            ret, frame = cap.read()
            if ret is not True:
                break
            cv2.imshow('Frame', frame)
            cv2.waitKey(700)
            go_back = False

        try:
            ball_pos[frame_pos] = pos
        except IndexError:
            ball_pos.append(pos)
            assert len(ball_pos) == frame_pos + 1

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    return ball_pos

if __name__ == '__main__':
    try:
        start_frame = int(sys.argv[1])
    except IndexError:
        start_frame = 0
    ball_pos = pd.read_csv('label.csv').apply(lambda row: (row['x'], row['y']), axis=1).tolist()
    ball_pos = label('output.mp4', start_frame, ball_pos)

    pd.DataFrame({'x': [v[0]  if v else None for v in ball_pos], 'y': [v[1]  if v else None for v in ball_pos]}).to_csv('label.csv')
