from tkinter import *
import cv2
from face import Face
from face_recognizer import FaceRecognizer

class GUI:
    """Set up and manage all the variables for the GUI interface"""

    def __init__(self):
        self.root = Tk()
        self.root.title("Noob Robot")

        self.face = None
        self.canvas = None

    def setup_widgets(self):
        """Set up all the parts of the GUI"""
        self._init_face()
        self._init_expression_tool()
        self._init_video_tool()

    def _init_face(self):
        """Set up the face"""
        canvasFrame = Frame(self.root)
        canvasFrame.grid(row=1, column=1, padx=5, pady=5)
        self.canvas = Canvas(canvasFrame, width=400, height=400)
        self.canvas.grid(row=1, column=1)

        self.face = Face(self.canvas)

    def _init_video_tool(self):
        videoFrame = Frame(self.root, bd=5, padx=10, pady=10)
        videoFrame.grid(row=2, column=1, padx=5, pady=5)
        self.runBtn = Button(videoFrame, text="Start", command=self.run_animation)
        self.runBtn.grid(row=1, column=1)

        self.stopBtn = Button(videoFrame, text="Stop")
        self.stopBtn.grid(row=1, column=2)

    def _init_expression_tool(self):
        expressionFrame = Frame(self.root, bd=5, padx=10, pady=10)
        expressionFrame.grid(row=1, column=2, padx=5, pady=5, sticky=N)
        expressionTitle = Label(expressionFrame, text="Expression Option", font="Arial 16 bold")
        expressionTitle.grid(row=0, column=1, padx=5, pady=5)

        self.smileBtn = Button(expressionFrame, text="Smile", command=self.face.smile)
        self.smileBtn.grid(row=1, column=1)
        self.sadBtn = Button(expressionFrame, text="Sad", command=self.face.sad)
        self.sadBtn.grid(row=2, column=1)
        self.grinBtn = Button(expressionFrame, text="Grin", command=self.face.grin)
        self.grinBtn.grid(row=3, column=1)

        self.smileBtn = Button()

    def run_animation(self):
        rc = FaceRecognizer(self.root, self.canvas)
        rc.start()

    def run_face_animation(self):
        """
        This callback for the Start button moves the eyeball and change the facial emotion
        :return:
        """
        faceCascade = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_alt0.xml")
        eye1Cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_eye1.xml")

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in face_rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                roi_gray = gray[y: y + h, x: x + w]
                roi_color = frame[y:y + h, x:x + w]

                eye = eye1Cascade.detectMultiScale(roi_gray)

                for (ex, ey, ew, eh) in eye:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 3)

            cv2.imshow("FD", frame)

            v = cv2.waitKey(20)
            c = chr(v & 0xFF)
            if c == 'q':
                break

    def run(self):
        """Start the whole GUI"""
        self.root.mainloop()


def run():
    gui = GUI()
    gui.setup_widgets()
    gui.run()


if __name__ == "__main__":
    run()
