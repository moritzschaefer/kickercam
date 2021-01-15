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

    lower_b = np.array([0,0,50])
    upper_b = np.array([255,100,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_b, upper_b)
    # Bitwise-AND mask and original image
    frame = cv2.bitwise_and(frame,frame, mask= mask)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale=1
    frame = cv2.resize(frame,(int(1280*scale),int(720*scale)))
    return frame

def try_tracking():

    # Set up tracker.
    # Instead of MIL, you can also use

    # Read video
    video = cv2.VideoCapture("dataset/v3.h265")

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
        cv2.resizeWindow('Tracking', 1280, 720)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

        elif k == ord('b'):
            bbox = cv2.selectROI(frame, False)

            tracker, tracker_type = get_tracker(Track_ID)
            # Initialize tracker with first frame and bounding box
            _ = tracker.init(frame, bbox)

            _ , bbox = tracker.update(frame)

def get_optical_flow_edges():
    cap = cv2.VideoCapture("dataset/v2.h265")
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    for i in range(540):
        ok, frame = cap.read()
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    bbox = cv2.selectROI(frame, False)
    print(bbox)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    print(p0.shape)
    p0[-1] = [[bbox[0],bbox[1]]]
    p0[-2] = [[bbox[0]+bbox[2],bbox[1]+bbox[3]]]
    print(p0)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while (1):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

def get_optical_flow_all():
    cap = cv2.VideoCapture("dataset/v4.h265")
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    while (1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2', bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', bgr)
        prvs = next
if __name__ == '__main__':
    #get_optical_flow_edges()
    try_tracking()