import cv2


def load_camera():
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)60/1  ! nvvidconv flip-method=0   !  appsink", cv2.CAP_GSTREAMER)
    #cap = cv2.VideoCapture("nvarguscamerasrc ! width={width}, height={height}, nvvidconv flip-method=0   !  appsink".format(width=640, height=480), cv2.CAP_GSTREAMER)
    # Capture frame-by-frame
    if not cap.isOpened():
        raise RuntimeError("cap is not open")
    return cap


    #while True:
        #ret, frame = cap.read()
        ##gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame',frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
#
    #cap.release()
    #cv2.destroyAllWindows()
