They data set shows several foosball matches (some realistic ones, some for demonstrational purposes) with the following features

- moving ball
- ocluded ball (by players, sometimes by hands)
- motion blur of the ball at strong shots
- different light conditions
- different camera perspectives (all from the top)
- different frame rates

There are 4 video files called v1 to v4. v1 was recorded using a GoPro camera with fish-eye perspective and a normal framerate. The others were recorded using a Waveshare IMX219-160 camera connected to a Nvidia Jetson Nano with a high frame rate (120FPS). Note that due to frame-skipping during the recording, there are a couple of "jumps" in the videos v2,v3 and v4 (not many though).

The command to record the videos v2, v3 and v4 is:

gst-launch-1.0 nvarguscamerasrc !   'video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,   format=(string)NV12, framerate=(fraction)120/1' ! nvv4l2h265enc   bitrate=8000000 ! h265parse ! qtmux ! filesink   location=<filename> -e

The video files of the data set can be downloaded here: https://polybox.ethz.ch/index.php/s/H662TzwhOz4xkGJ

The ball positions were labeled manually, as described in doc/LABELING.md.
