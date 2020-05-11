import tkinter as tk
import threading
import dlib
import cv2
import numpy as np
from mood_tracker import MoodTracker
import random
import time


DEBUG = True
WAIT_TIME = 1


class Coord:
    def __init__(self, x, y, w, h):
        """
        This is a class to keep track of a rectangle in the coordinate
        :param x:
        :param y:
        :param w: width
        :param h: height
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_pos(self):
        """
        Get the position of the rect
        :return: x0, y0, x1, y1
        """
        return self.x, self.y, self.x + self.w, self.y + self.h

    def __bool__(self):
        return self.x == -1 and self.y == -1 and self.w == -1 and self.h == -1

    def __str__(self):
        return f"{self.x, self.y, self.w, self.h}"


class FaceRecognizer(threading.Thread):
    """
    A thread for keeping track of the facial landmarks and update the face expression on the canvas
    """
    def __init__(self, root: tk.Tk, canvas: tk.Canvas, run_text: tk.StringVar, run_btn: tk.Button, chat_text: tk.StringVar):
        super().__init__()
        self.root = root
        self.canvas = canvas
        self.run_text = run_text
        self.run_btn = run_btn

        self.chat_text = chat_text
        self.chat_time = None

        predictor_path = "predictors/shape_predictor_68_face_landmarks.dat"

        self.detector = dlib.get_frontal_face_detector()  # Detector for the frontal face
        self.predictor = dlib.shape_predictor(predictor_path)  # Predictor for facial landmark

        self.color_green = (0, 255, 0)
        self.line_width = 3

        self.face_pos = Coord(70, 70, 350 - 70, 350 - 70)
        self.det_pos = None
        self.lms = None

        self._stop_event = threading.Event()
        self._is_running = True

        self.mt = MoodTracker()

    def transform_pt(self, x, y) -> tuple:
        """
        Transform a point in the image to a point in the canvas
        :param x:
        :param y:
        :return: new x,y
        """
        x -= self.det_pos.x
        y -= self.det_pos.y

        x = x / self.det_pos.w * self.face_pos.w + self.face_pos.x
        y = y / self.det_pos.h * self.face_pos.h + self.face_pos.y

        return int(x), int(y)

    def get_feature_pos(self, num1, num2):
        """
        Get 2 coordinates of the landmark of the given number
        :param num1:
        :param num2:
        :return:
        """
        if self.lms is None:
            raise ValueError("No landmarks")
        l_pt = self.lms[num1]
        x, y = (l_pt[0], l_pt[1])
        r_pt = self.lms[num2]
        x2, y2 = (r_pt[0], r_pt[1])
        return x, y, x2, y2

    # Functions to update the face features on the canvas
    def set_mouth(self):
        upper_mouth = []
        for i in (48, 63, 54):
            c_pt = self.lms[i]
            x, y = (c_pt[0], c_pt[1])
            x, y = self.transform_pt(x, y)
            upper_mouth.extend((x, y))

        x, y, x1, y1, x2, y2 = upper_mouth

        self.canvas.delete('upper_mouth')
        self.canvas.create_line(x, y, x1, y1, x2, y2, width=5, smooth='true', tags='upper_mouth')

        lower_mouth = []
        for i in (48, 57, 54):
            c_pt = self.lms[i]
            x, y = (c_pt[0], c_pt[1])
            x, y = self.transform_pt(x, y)
            lower_mouth.extend((x, y))

        x, y, x1, y1, x2, y2 = lower_mouth

        self.canvas.delete('lower_mouth')
        self.canvas.create_line(x, y, x1, y1, x2, y2, width=5, smooth='true', tags='lower_mouth')

    def transform_eye(self, x0, y0, x1, y1):
        """
        Make the eye bigger and more round
        :param x0:
        :param y0:
        :param x1:
        :param y1:
        :return:
        """
        r = (x1 - x0) / 2 + 10
        return int(x0)-10, int(y0 - r), x1+10, int(y1 + r)

    def set_eyeballs(self):
        x0, y0, x1, y1 = self.get_feature_pos(36, 39)
        cx, cy = (x1 + x0) / 2, (y0 + y1) / 2
        cx, cy = self.transform_pt(cx, cy)
        # r = ((x1 - x0) / 2 + 10)
        self.canvas.delete("leftBall")
        self.canvas.create_oval(cx-10, cy-10, cx+10, cy+10, fill="black", tag='leftBall')

        x0, y0, x1, y1 = self.get_feature_pos(42, 45)
        cx, cy = (x1 + x0) / 2, (y0 + y1) / 2
        cx, cy = self.transform_pt(cx, cy)
        self.canvas.delete("rightBall")
        r = (x1 - x0) / 2 + 10
        self.canvas.create_oval(cx-10, cy-10, cx + 10, cy + 10, fill="black", tag='rightBall')

    def set_eyes(self):
        x0, y0, x1, y1 = self.get_feature_pos(36, 39)
        x0, y0, x1, y1 = self.transform_eye(x0, y0, x1, y1)
        x0, y0 = self.transform_pt(x0, y0)
        x1, y1 = self.transform_pt(x1, y1)
        self.canvas.delete('left')
        self.canvas.create_oval(x0, y0, x1, y1, fill='white', tags='left', width=5, outline="#ED950D")

        x0, y0, x1, y1 = self.get_feature_pos(42, 45)
        x0, y0, x1, y1 = self.transform_eye(x0, y0, x1, y1)
        x0, y0 = self.transform_pt(x0, y0)
        x1, y1 = self.transform_pt(x1, y1)
        self.canvas.delete('right')
        self.canvas.create_oval(x0, y0, x1, y1, fill='white', tags='right', width=5, outline="#ED950D")

    def set_eyebrows(self):
        x0, y0, x1, y1 = self.get_feature_pos(18, 20)
        x0, y0 = self.transform_pt(x0, y0)
        x1, y1 = self.transform_pt(x1, y1)
        self.canvas.delete("leftBrow")
        self.canvas.create_line(x0, y0, x1, y1, width=5, tags='leftBrow')

        x0, y0, x1, y1 = self.get_feature_pos(23, 25)
        x0, y0 = self.transform_pt(x0, y0)
        x1, y1 = self.transform_pt(x1, y1)
        self.canvas.delete("rightBrow")
        self.canvas.create_line(x0, y0, x1, y1, width=5, tags='rightBrow')

    def set_nose(self):
        x0, y0, x1, y1 = self.get_feature_pos(28, 33)
        x0, y0 = self.transform_pt(x0, y0)
        x1, y1 = self.transform_pt(x1, y1)
        self.canvas.delete("noseBridge")

        noseAlign = (x0 - self.face_pos.x) / self.face_pos.w
        if noseAlign > 0.52:
            x2, y2 = self.lms[32]
        elif noseAlign < 0.48:
            x2, y2 = self.lms[34]
        else:
            x2, y2 = self.lms[33]
        x2, y2 = self.transform_pt(x2, y2)
        self.canvas.delete("noseTip")
        self.canvas.create_line(x0, y0, x1, y1, x2, y2, width=5, tags='noseTip', smooth='true')

    def set_face(self):
        """
        Update all features on the face
        :return:
        """
        self.set_mouth()
        self.set_eyes()
        self.set_eyeballs()
        self.set_eyebrows()
        self.set_nose()

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
            det = dets[0]  # Find the essential face
            x0, y0, x1, y1 = det.left(), det.top(), det.right(), det.bottom()
            # Draw rectangle around the face
            cv2.rectangle(img, (x0, y0), (x1, y1), self.color_green, self.line_width)
            self.lms = np.array([[p.x, p.y] for p in self.predictor(img, det).parts()])

            x0 = self.lms[0][0]
            y0 = self.lms[19][1]
            x1 = self.lms[16][0]
            y1 = self.lms[8][1]

            y0 -= (y1-y0) * 0.2     # Scale the height to include the forehead

            self.det_pos = Coord(x0, y0, x1-x0, y1-y0)

            for idx, point in enumerate(self.lms):
                pos = (point[0], point[1])
                cv2.circle(img, pos, 1, color=(0, 255, 255))
            return img

    def handle_mood(self, face):
        self.mt.update_emotion_value(face)
        if self.mt.mood_value > 100:
            self.set_chat("Happy")
            self.mt.mood_value = 0
        elif self.mt.mood_value < -100:
            self.set_chat("Sad")
            self.mt.mood_value = 0

    def set_chat(self, mood):
        jokes = (
            "How does a rabbi make coffee? Hebrews it!",
            "Rest in peace boiling water. You will be mist!",
            "How do you throw a space party? You planet!",
            "I ate a clock yesterday, it was very time-consuming",
            "Have you played the updated kids' game? I Spy With My Little Eye . . . Phone.",
            "If we shouldn't eat at night, why do they put a light in the fridge?"
        )

        stoics = (
            "Waste no more time arguing what a good man should be. Be One.",
            "It never ceases to amaze me: we all love ourselves more than other people, but care more about their opinion than our own.",
            "The best revenge is not to be like your enemy.",
            "If it is not right, do not do it, if it is not true, do not say it.",
            "Be tolerant with others and strict with yourself."
        )

        if self.chat_time and time.time() - self.chat_time > 5:
            self.chat_time = None
            self.chat_text.set("Hello again!")
            self.root.update()
            return

        if self.chat_time:
            return

        if mood == "Happy":
            text = random.choice(jokes)
        elif mood == "Sad":
            text = random.choice(stoics)
        else:
            raise ValueError("Invalid argument value of mood")
        self.chat_text.set(text)
        self.root.update()
        self.chat_time = time.time()

    def run(self):
        """
        Recognize the facial landmark on the video capture, and update the face on the canvas
        This will be called after the thread starts
        :return:
        """
        cam = cv2.VideoCapture(0)
        while not self._stop_event.isSet():
            ret_val, img = cam.read()
            annotated_img = self.annotate_face(img)

            if img is None:
                break

            # Update the chat according to the mood
            face, coord = self.mt.find_faces(img)
            self.handle_mood(face)

            try:
                self.run_text.set("Press 'Esc' to Stop")
                self.run_btn["state"] = "disabled"
                self.set_face()
                self.root.update()

                self.handle_mood(face)
            except ValueError:
                print("Nothing gets updated")

            cv2.imshow('my webcam', annotated_img)
            if cv2.waitKey(WAIT_TIME) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
        self.run_text.set("Start Animation")
        self.run_btn["state"] = "normal"
        self.root.update()
        print("The video capture ends")


    def stop(self):
        self._is_running = False
        cv2.destroyAllWindows()
        self._stop_event.set()

    def join(self, timeout=None):
        cv2.destroyAllWindows()
        self._stop_event.set()
        threading.Thread.join(self, timeout)

