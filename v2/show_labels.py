import math

import cv2
import pandas as pd

RESIZE = 1  # set to 1 to disable scaling
SHOW_BALL_LABEL = True

# 6800 -> 7250
def main(video_fn, label_fn):

    delay = 25
    cap = cv2.VideoCapture(video_fn)

    # Check if camera opened successfully
    if cap.isOpened() == False: 
        print("Error opening video stream or file")

    cv2.namedWindow('Frame') 

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
            cv2.circle(frame, point, math.ceil(9*RESIZE), (0, 0, 255), math.ceil(8*RESIZE))

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
    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main('output.mp4', 'label.csv')
