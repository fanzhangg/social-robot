import cv2
import numpy as np
from os import path
import joblib
import dlib
import math

"""
This is the demo using webcam to recognize the emotion of a face using a model trained with dlib and sklearn
"""

# TODO: WIP

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "..\\predictors\\shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file


detPath = path.abspath("../haarcascades/haarcascade_frontalface_default.xml")
faceDet = cv2.CascadeClassifier(detPath)
detPath2 = path.abspath("../haarcascades/haarcascade_frontalface_alt2.xml")
faceDet_two = cv2.CascadeClassifier(detPath2)
detPath3 = path.abspath("../haarcascades/haarcascade_frontalface_alt0.xml")
faceDet_three = cv2.CascadeClassifier(detPath3)
detPath4 = path.abspath("../haarcascades/haarcascade_frontalface_alt_tree.xml")
faceDet_four = cv2.CascadeClassifier(detPath4)
font = cv2.FONT_HERSHEY_SIMPLEX
UpperLeftCornerOfText = (10, 30)
SecondUpperLeftCornerOfText = (100, 30)
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2
emotion = ["Happy", "Sad"]


def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))

    if len(detections) < 1:
        return None
    else:
        return landmarks_vectorised


def find_faces(image):
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

def predict(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_image = clahe.apply(gray)
    landmarks = get_landmarks(clahe_image)
    np.array(landmarks)

def showWebCamAndRun(model):
    """
    Shows webcam image, detects faces and its emotions in real time and draw emoticons over those faces.
    :param model: Learnt emotion detection model.
    :param window_size: Size of webcam image window.
    :param window_name: Name of webcam image window.
    """

    cam = cv2.VideoCapture(0)
    while True:

        ret, frame = cam.read()
        if frame is None:
            break

        f, (x, y, w, h) = find_faces(frame)
        if f is None:
            cv2.putText(frame, "Please put your face in front of the webcam!",
                                    UpperLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
            continue
        prediction = model.predict(f)
        confidence = 0
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(prediction),
                    UpperLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)


        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
    # cleanup the camera and close any open windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # load model
    model = joblib.load("emotion_model.pkl")
    #fisher_face.read(p2)

    # use learnt model
    window_name = 'WEBCAM (press q to exit)'
    showWebCamAndRun(model)
