# kickercam

This repository contains code for several tasks related to, ball tracking and slow-motion replay of a foosball game.
The project aims to realize the following tasks:

- Recording and labeling of ball position
- Algorithm to detect the position of a ball in the game
- Camera live view with goal detection and slow-motion replay of the shot

New version in "jetson". The rest sux.
Jetson IP: 192.168.178.111, VNC password: password (connect with tightvnc client, or similar).

Foosball (Kicker, Tablesoccer) live camera with goal detection and slow-motion replay

python3 main2.py starts the camera and outputs to a Monitor. Replay is Implemented but currently not used.  

# Installation

To run the labeler scripts the following libraries are required:

- python3 (3.7 or higher)
- python3-opencv
- pandas

For the modeling part:

- pytorch
- torchvision
- cudatoolkit


You can use conda/mamba:

mamba install pandas opencv pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c conda-forge

# Dataset

More information in doc/DATASET.md

# Hardware

Nvidia Jetson nano (4GB RAM) with Waveshare IMX219-160 camera.
Autostart was setup using "Startup Applications"
# TODOs

- For now, jetson/load_camera.py is used to display the kickercam in fullscreen mode. This should be merged with video_reader.py (to make use of the ringbuffer)
- The python project is not setup properly. This has to be fixed to make relative imports work (e.g. video_reader.py)
- ball-inference is not yet used at all
