# kickercam

New version in "jetson". The rest sux.
Jetson IP: 192.168.178.111, VNC password: password (connect with tightvnc client, or similar).


Foosball (Kicker, Tablesoccer) live camera with goal detection and slow-motion replay

python3 main2.py starts the camera and outputs to a Monitor. Replay is Implemented but currently not used.  

# train

python3 classifier.py

trains and saves the model. Note that classifier.py is also used for predictions (see main.py)


# TODOs

If you apply all these TODOs it will work perfectly!

## Add heat sink

It's getting to hot with the analysis

Add a heat sink to the raspberry chip (we should still have some). Furthermore add a small fan!


## adjust cam and cut out field area

Center the image. Use hotglue on the camera and adjust it perfectly. make sure the pi sits still as well

There is an option to cut out (and zoom) an area of the capture frame.
Look here for zoom:
https://picamera.readthedocs.io/en/release-1.13/api_camera.html?highlight=zoom

## After cutting the stream as indicated above, it should be possible to increase the frame rate to about 42 FPS (which is enough for a cool slowmo already).

Increase the FPS until the program crashes

# Use different resolutions

We can use the full 1600x1200 for the preview (and replay) and use 600x480 for the image processing to speed up everything

# video formats

Right now we use h264 for the circular replay buffer and mjpeg for the analysis. It's probably faster to use the same codec for both (we NEED mjpeg for the analysis).

## use real multiprocessing

Right now only multithreading is used (python does not have real multithreading and only and core is used at a time). Use multiprocessing in order to use all kernels...

##


# Dataset preprocessing

Data was generated using Jetson with blabla camera and command gst-launch...

Afterwards, files were post processed using 

ffmpeg -i trainingdata/third.h265 -err_detect aggressive -fflags discardcorrupt -vcodec h264 -crf 21 third.avi

in order to drop corrupted frames, recorded by the camera/jetson.
