import cv2
import glob
import random
import numpy as np
from os import path, listdir

emotions = ["Happy", "Sad"]  # Emotion list
fishface = cv2.face.FisherFaceRecognizer_create()  # Initialize fisher face classifier
data = {}


def get_files(emotion):
    """Define function to get file list, randomly shuffle it and split 80/20"""

    p = path.abspath(f"./source_images/dataset/{emotion}/")
    fileNames = listdir(p)
    files = []
    for name in fileNames:
        files.append(f"{p}/{name}")

    random.shuffle(files)
    training = files[:int(len(files) * 0.9)]  # get first 90% of file list
    prediction = files[-int(len(files) * 0.1):]  # get last 10% of file list
    return training, prediction


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-1
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            training_data.append(gray)  # append image array to training data list
            training_labels.append(emotions.index(emotion))
        for item in prediction:  # repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    print("training fisher face classifier")
    print("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    print("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    numberDone = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
        numberDone += 1
    print(training_data)
    return (100 * correct) / (correct + incorrect)


# Now run it
metascore = []
run_recognizer()
# for i in range(10):
#     correct = run_recognizer()
#     print("got", correct, "percent correct!")
#     metascore.append(correct)
# fishface.save('model/emotion_detection_model.xml')
# print("\n\nend score:", np.mean(metascore), "percent correct!")
