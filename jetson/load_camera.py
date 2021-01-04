import cv2

class CameraLoader:
    def __init__(self, width=1280, height=720,fps=60, filename = None):
        if filename:
            self.cap = cv2.VideoCapture(filename)
        else:
            # if True:
            self.cap = cv2.VideoCapture(
                f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, format=(string)NV12, framerate=(fraction){fps}/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
            # gst_str = ('nvarguscamerasrc ! '
               # 'video/x-raw(memory:NVMM), '
               # 'width=(int)1920, height=(int)1080, '
               # 'format=(string)NV12, framerate=(fraction)30/1 ! '
               # 'nvvidconv flip-method=2 ! '
               # 'video/x-raw, width=(int){}, height=(int){}, '
               # 'format=(string)BGRx ! '
               # 'videoconvert ! appsink').format(width, height)
                # self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Camera is not open")

    def get_image(self):
        """
        returns: tuple, (cv2 return, image)
        """
        return self.cap.read()

    def __del__(self):
        self.cap.release()

    def display_image(self):
        ret, frame = self.cap.read()
        cv2.imshow('frame', frame)

if __name__ == '__main__':
    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cl = CameraLoader()
    while cl.cap.isOpened():
        cl.display_image()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.cap.release()

"""
self.cap = cv2.VideoCapture(
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int){}, height=(int){}, format=(string)NV12, framerate=(fraction){}/1  ! nvvidconv flip-method=0   !  appsink".format(width,height,fps),
            cv2.CAP_GSTREAMER)
"""

