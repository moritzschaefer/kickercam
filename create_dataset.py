import os

from goal_detector import GoalDetector


def main():
    try:
        os.mkdir('dataset')
    except OSError:
        pass
    gd = GoalDetector()
    videopath = './match1.h264'
    camera = cv2.VideoCapture(videopath)
    (grabbed, frame) = camera.read()
    while grabbed:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        obs, _ = gd.step(hsv)
        (grabbed, frame) = camera.read()



if __name__ == "__main__":
    main()
