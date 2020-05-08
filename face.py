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
        self.det_coord = Coord(0, 0, 0, 0)
        self.lms = []

        self.face_coord = Coord(70, 70, 350 - 70, 350 - 70)

        self._init_face_recognizer()
        self._init_face(canvas)

    def _init_face_recognizer(self):
        predictor_path = "predictors/shape_predictor_68_face_landmarks.dat"

        self.detector = dlib.get_frontal_face_detector()  # Detector for the frontal face
        self.predictor = dlib.shape_predictor(predictor_path)  # Predictor for facial landmark

        self.color_green = (0, 255, 0)
        self.line_width = 3

    def _init_face(self, canvas):
        self.canvas = canvas
        self.hasAnimation = False

        canvas.create_oval(70, 70, 350, 350, fill='#FEE230', width=5, outline="#ED950D")
        canvas.create_oval(110, 110, 200, 200, fill='white', tags='left', width=5, outline="#ED950D")
        canvas.create_oval(110 + 30, 110 + 30, 110 + 60, 110 + 60, fill="black", tag='leftBall')
        canvas.create_oval(310 - 90, 110, 310, 110 + 90, fill='white', tags='right', width=5, outline="#ED950D")
        canvas.create_oval(220 + 30, 110 + 30, 220 + 60, 110 + 60, fill="black", tag="rightBall")
        canvas.create_line(145, 250, 275, 250, width=5, tags='mouth')

    def _scale_pos(self, pos: Coord):
        ratio = self.face_coord.w / self.det_coord.w
        pos.w = pos.w * ratio
        pos.h = pos.h * ratio
        pos.x = self.face_coord.x
        pos.y = self.face_coord.y
        return pos

    def annotate_face(self, img):
        """
        Annotate the face rectangle and the face landmarks
        :param img: an image
        :return: the annotated face if a face is presented in the image, else return the original img
        """
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = self.detector(rgb_image)
        if len(dets) == 0:
            return img
        else:
            # TODO: Try to find out the most essential face rather tahn the first face by default
            det = dets[0]  # Find the essential face
            # Draw rectangle around the face
            w = det.right()-det.left()
            self.det_coord = (0, 0, w, w)
            cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), self.color_green, self.line_width)
            self.lms = np.matrix([[p.x, p.y] for p in self.predictor(img, det).parts()])
            for idx, point in enumerate(self.lms):
                pos = (point[0, 0], point[0, 1])
                # cv2.putText(im, str(idx), pos,
                #             fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                #             fontScale=0.4,
                #             color=(0, 0, 255))
                cv2.circle(img, pos, 1, color=(0, 255, 255))
            return img

    def get_mouth_pos(self)->Coord:
        l_pt = self.lms[48]
        x, y = (l_pt[0, 0], l_pt[0, 1])
        r_pt = self.lms[54]
        x2, y2 = (r_pt[0, 0], r_pt[0, 1])
        w, h = x2 - x, y2 - y
        return Coord(x, y, w, h)

    def set_mouth(self):
        x0, y0, x1, y1 = self.get_mouth_pos().get_pos()
        print(f"Update mouth: {x0}, {y0}, {x1}, {y1}")
        self.canvas.delete('mouth')
        self.canvas.create_line(x0, y0, x1, y1, width=5, tags='mouth')

    def set_face(self):
        self.set_mouth()

    def run(self):
        cam = cv2.VideoCapture(0)
        while True:
            ret_val, img = cam.read()
            img = self.annotate_face(img)

            self.set_face()

            cv2.imshow('my webcam', img)
            if cv2.waitKey(10) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()

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