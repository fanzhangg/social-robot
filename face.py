#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in a webcam stream using OpenCV.
#   It is also meant to demonstrate that rgb images from Dlib can be used with opencv by just
#   swapping the Red and Blue channels.
#
#   You can run this program and see the detections from your webcam by executing the
#   following command:
#       ./opencv_face_detection.py
#
#   This face detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.  This type of object detector
#   is fairly general and capable of detecting many types of semi-rigid objects
#   in addition to human faces.  Therefore, if you are interested in making
#   your own object detectors then read the train_object_detector.py example
#   program.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import sys
import dlib
import cv2
import numpy as np

class Coord:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_pos(self):
        return self.x, self.y, self.x + self.w, self.y + self.h


class Face:

    def __init__(self, canvas):
        self._init_face(canvas)

    def _init_face(self, canvas):
        self.canvas = canvas

        canvas.create_oval(70, 70, 350, 350, fill='#FEE230', width=5, outline="#ED950D")
        canvas.create_oval(110, 110, 200, 200, fill='white', tags='left', width=5, outline="#ED950D")
        canvas.create_oval(110 + 30, 110 + 30, 110 + 60, 110 + 60, fill="black", tag='leftBall')
        canvas.create_oval(310 - 90, 110, 310, 110 + 90, fill='white', tags='right', width=5, outline="#ED950D")
        canvas.create_oval(220 + 30, 110 + 30, 220 + 60, 110 + 60, fill="black", tag="rightBall")
        canvas.create_line(145, 250, 275, 250, width=5, tags='mouth')

    def smile(self):
        self.canvas.delete('mouth')
        self.canvas.create_arc(125, 225, 275, 275, extent=-180, width=5, fill='white', tags='mouth')

    def sad(self):
        self.canvas.delete('mouth')
        self.canvas.create_arc(125, 250, 275, 300, extent=180, width=5, fill='white', tags='mouth')

    def wink(self):
        self.canvas.delete('mouth')
        self.canvas.create_line(125, 250, 275, 250, width=5, tags='mouth')

    def grin(self):
        self.canvas.delete('mouth')
        self.canvas.create_line(125, 250, 200, 250, 275, 215, width=5, smooth='true', tags='mouth')


if __name__ == "__main__":
    pass