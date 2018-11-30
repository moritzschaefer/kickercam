#!/usr/bin/python3 -u
"""
Record last n seconds of picamera in ring buffer and replay with vnc from correct position (all in memory)
"""

import picamera
from time import sleep
import sys
import subprocess


def record_and_replay(n):
    recording_time = n
    slowtime = "0.5"
    try:
        # Initialize camera
        camera = picamera.PiCamera()
        #camera.resolution = (1600, 900)
        camera.resolution = (1640, 922)
        #camera.resolution = (1920, 1080)
        camera.zoom = (0.04, 0.07, 0.83, 0.83)
        camera.framerate  = 40
        # Start vlc in fullscreen and save piperef
        #cmdline = ['vlc', '--demux','mjpeg', '--fullscreen', '--rate', slowtime, '-']

            
        # Wait till everything is ready and start preview
        sleep(1)
        camera.start_preview(fullscreen=False, window = (0,0,1920,1080))
        
        # Start recording
        ring_buffer = picamera.PiCameraCircularIO(camera, seconds=recording_time*2)
        camera.start_recording(ring_buffer,'h264')
                
        while True:
            
            sleep(10)
            #For ratio setting
            #x,y,h,w = map(float,input("values").split())
            #camera.zoom = (x,y,h,w)
            
        while False:
            
            # Write to vlc stdin
            #cmdline = ['vlc', '--demux','h264', '-vvv', '--fullscreen', '-']
            cmdline = ['vlc', '--demux','h264', '--fullscreen', '-']
            player = subprocess.Popen(cmdline, stdin=subprocess.PIPE)
            camera.stop_recording()
            ring_buffer.copy_to(player.stdin, seconds=recording_time)
            # Now stop preview
            camera.stop_preview()
            # Wait for recording to be played
            sleep(recording_time*3)
            camera.start_preview(fullscreen=False, window = (0,0,1920,1080))
            ring_buffer = picamera.PiCameraCircularIO(camera, seconds=recording_time*2)
            player.kill()
            camera.start_recording(ring_buffer,'h264')
            
    except:
        camera.stop_recording()
        camera.stop_preview()
        player.kill()
        

if __name__ == "__main__":
    record_and_replay(5)
    