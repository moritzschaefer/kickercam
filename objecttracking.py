import cv2
import sys
import numpy as np
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def get_tracker(id):

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[id%8]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    return tracker, tracker_type
Track_ID = 2


def get_image(video):
    ok, frame = video.read()

    assert ok, "Could not read video"
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_b = np.array([0,0,80])
    upper_b = np.array([255,100,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_b, upper_b)
    # Bitwise-AND mask and original image
    frame = cv2.bitwise_and(frame,frame, mask= mask)
    return frame
if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use

    # Read video
    video = cv2.VideoCapture("dataset/v2.h265")

    # Exit if video not opened.
    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print
        'Cannot read video file'
        sys.exit()
    for i in range(500):
        ok, frame = video.read()
    frame = get_image(video)

    tracker, tracker_type = get_tracker(Track_ID)
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        frame = get_image(video)

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

        elif k == ord('b'):
            bbox = cv2.selectROI(frame, False)

            tracker, tracker_type = get_tracker(Track_ID)
            # Initialize tracker with first frame and bounding box
            _ = tracker.init(frame, bbox)

            _ , bbox = tracker.update(frame)
