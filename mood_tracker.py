import cv2
import numpy
from os import path

detPath = path.abspath("./haarcascades/haarcascade_frontalface_default.xml")
faceDet = cv2.CascadeClassifier(detPath)
detPath2 = path.abspath("./haarcascades/haarcascade_frontalface_alt2.xml")
faceDet_two = cv2.CascadeClassifier(detPath2)
detPath3 = path.abspath("./haarcascades/haarcascade_frontalface_alt0.xml")
faceDet_three = cv2.CascadeClassifier(detPath3)
detPath4 = path.abspath("./haarcascades/haarcascade_frontalface_alt_tree.xml")
faceDet_four = cv2.CascadeClassifier(detPath4)
font = cv2.FONT_HERSHEY_SIMPLEX
UpperLeftCornerOfText = (10, 30)
SecondUpperLeftCornerOfText = (100, 30)
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2
emotion = ["Happy", "Sad"]


class MoodTracker:
    def __init__(self):
        """
        This is a class to keep track of the user's mood based on the history of its emotion
        """
        # load model
        p1 = path.abspath(f"./model/emotion_detection_model.xml")
        p2 = path.abspath(f"./model/emotion_detection_model_large.xml")
        self.model = cv2.face.FisherFaceRecognizer_create()
        self.model.read(p2)

        self.mood_value = 0

    def find_faces(self, image):
        face = faceDet.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
        # Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""
        # Cut and save face
        out = None
        x, y, w, h = 0, 0, 0, 0
        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            image = image[y:y + h, x:x + w]  # Cut the frame to size
            out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            out = cv2.resize(out, (300, 300))  # Resize face so all images have same size
        return out, (x, y, w, h)

    def update_emotion_value(self, f):
        """
        Predict the emotion of the user, update the mood value
        :param model: Learnt emotion detection model.
        :param window_size: Size of webcam image window.
        :param window_name: Name of webcam image window.
        """
        if f is None:
            return False
        prediction = self.model.predict(f)
        confidence = 0
        if cv2.__version__ != '3.1.0':
            confidence = str(prediction[1])
            prediction = prediction[0]
        print("Confidence:", confidence, "\tprediction:", prediction, "\tmood:", self.mood_value)
        if emotion[prediction] == "Happy":
            self.mood_value += float(confidence)
        else:
            self.mood_value -= float(confidence)
        return self.mood_value

    def run(self):
        cam = cv2.VideoCapture(0)
        while True:

            ret, frame = cam.read()
            if frame is None:
                break

            f, (x, y, w, h) = self.find_faces(frame)
            if f is None:
                cv2.putText(frame, "Please put your face in front of the webcam!",
                            UpperLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                continue
            prediction = self.model.predict(f)
            confidence = 0
            if cv2.__version__ != '3.1.0':
                confidence = str(prediction[1])
                prediction = prediction[0]

            emo = self.update_emotion_value(f)
            self.mood_value += emo
            print(f"{emo}\t{self.mood_value}")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion[prediction],
                        UpperLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.putText(frame, confidence,
                        SecondUpperLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            cv2.imshow("hello", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break
        # cleanup the camera and close any open windows
        cv2.destroyAllWindows()


if __name__ == '__main__':
    mt = MoodTracker()
    mt.run()

