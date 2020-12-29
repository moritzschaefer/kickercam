import load_camera
import cv2
import time

def capture_dataset(filename='data.mp4',duration = 240):
    width, height = 1280, 720
    filename = "start{}".format(int(time.time()))+filename
    cam = load_camera.Camera_Loader()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 15.0, (width,height))
    # record video
    start_time = time.time()
    while time.time()-start_time < duration:
        ret, frame = cam.get_image()
        if ret:
            out.write(frame)
            #cv2.imshow('Video Stream', frame)
        else:
            print("We did not get an frame at time: ", time.time())7
            break

    del cam
    out.release()
    cv2.destroyAllWindows()



def load_video(video_fn, target_width, target_height, normalize_values=True):

    X = np.ndarray((y.shape[0], target_width,target_height, 3))

    cap = cv2.VideoCapture(video_fn)
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    for frame_i in range(X.shape[0]):
        ret, frame = cap.read()
        if ret is not True:
            raise RuntimeError('no more frames in video -.-')

        frame = cv2.resize(frame, (target_width,target_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = np.asarray(frame)
        frame = frame.astype("float")
        if normalize_values:
            frame /= 255.0

        X[frame_i, :, :, :] = np.transpose(frame, axes=[1,0,2])

    return X, y


if __name__ == "__main__":
    capture_dataset()
