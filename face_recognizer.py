import tkinter as tk
import threading
import dlib
import cv2
import numpy as np

debug = True


class Coord:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_pos(self):
        return self.x, self.y, self.x + self.w, self.y + self.h

    def __bool__(self):
        return self.x == -1 and self.y == -1 and self.w == -1 and self.h == -1


class FaceRecognizer(threading.Thread):
    def __init__(self, root: tk.Tk, canvas: tk.Canvas):
        super().__init__()
        self.root = root
        self.canvas = canvas

        predictor_path = "predictors/shape_predictor_68_face_landmarks.dat"

        self.detector = dlib.get_frontal_face_detector()  # Detector for the frontal face
        self.predictor = dlib.shape_predictor(predictor_path)  # Predictor for facial landmark

        self.color_green = (0, 255, 0)
        self.line_width = 3

        self.face_pos = Coord(70, 70, 350 - 70, 350 - 70)
        self.det_pos = Coord(-1, -1, -1, -1)
        self.lms = None

    def scale_points(self, pts: np.array) -> np.array:
        # xs, ys = pts.T      # x and y values of the given points
        # cx, cy = np.mean(xs), np.mean(ys)   # center
        if self.det_w is None:
            raise ValueError("No det width")
        # xr, yr = self.det_w / self.face_w, self.det_w * self.face_w     # Ratio of center points
        # ncx, ncy = cx / xr + , cy / yr   # New center point
        # vxs, vys = xs - cx, ys - cy     # Vectors from the center points
        # vxs, yxs = vxs / xr, vys / yr   # New vectors from the center points
        # npts = np.array(list(zip(vxs + ncx, yxs + ncy)))     # New points
        # if debug:
        #     print(f"Scaled points: {npts}")
        # return npts
        pts = pts / self.face_pos.w

    def transform_pt(self, x, y) -> tuple:
        """
        Transform a point in the image to a point in the canvas
        :param x:
        :param y:
        :return:
        """
        x -= self.face_pos.x
        y -= self.face_pos.y

        x * self.face_pos.w / self.det_pos.w + self.face_pos.x
        y * self.face_pos.h / self.det_pos.h + self.face_pos.y

        return int(x), int(y)

    def get_mouth_pos(self):
        if self.lms is None:
            raise ValueError("No landmarks")
        l_pt = self.lms[48]
        x, y = (l_pt[0], l_pt[1])
        r_pt = self.lms[54]
        x2, y2 = (r_pt[0], r_pt[1])
        if debug:
            print(f"Get mouse points: {x, y, x2, y2}")
        return x, y, x2, y2

    def set_mouth(self):
        x, y, x2, y2 = self.get_mouth_pos()
        x, y = self.transform_pt(x, y)
        x2, y2 = self.transform_pt(x2, y2)
        self.canvas.delete('mouth')
        self.canvas.create_line(x, y, x2, y2, width=5, tags='mouth')

    def set_face(self):
        self.set_mouth()

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
            self.det_pos = Coord(det.left(), det.top(), det.right()-det.left(), det.bottom()-det.top())
            cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), self.color_green, self.line_width)
            self.lms = np.array([[p.x, p.y] for p in self.predictor(img, det).parts()])
            for idx, point in enumerate(self.lms):
                pos = (point[0], point[1])
                # cv2.putText(im, str(idx), pos,
                #             fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                #             fontScale=0.4,
                #             color=(0, 0, 255))
                cv2.circle(img, pos, 1, color=(0, 255, 255))
            return img

    def run(self):
        cam = cv2.VideoCapture(0)
        while True:
            ret_val, img = cam.read()
            img = self.annotate_face(img)

            try:
                self.set_face()
                self.root.update()
            except ValueError:
                print("Nothing gets updated")

            cv2.imshow('my webcam', img)
            if cv2.waitKey(10) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
