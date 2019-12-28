import cv2
import pandas as pd


def main(video_fn, label_fn):
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
    while cap.isOpened() and frame_pos + 2 < len(df):
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is not True:
            break

        pos = df.loc[frame_pos]
        if pos['x'] >= 0:
            cv2.circle(frame, (pos['x'], pos['y']), 9, (0, 0, 255), 8)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        key = cv2.waitKey(int(25)) & 0xFF
        # print(key)

        if key == ord('q'):
            break
    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main('output.mp4', 'label.csv')
