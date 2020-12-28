import cv2

class Camera_Loader:
    def __init__(self, width=1280, height=720,fps=60, filename = None):
        if filename:
            self.cap = cv2.VideoCapture(filename)
        else:
            self.cap = cv2.VideoCapture(
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int){}, height=(int){}, format=(string)NV12, framerate=(fraction){}/1  ! nvvidconv flip-method=0   !  appsink".format(width,height,fps),
            cv2.CAP_GSTREAMER)
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

