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

# Dataset

More information in doc/DATASET.md

# Hardware

Nvidia Jetson nano (4GB RAM) with Waveshare IMX219-160 camera.
