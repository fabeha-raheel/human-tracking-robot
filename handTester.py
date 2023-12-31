#!/usr/bin/env python3

# %%
# To Install Libraries
# pip3 install --no-cache-dir opencv-python mediapipe

# Importing Libraries
import string
import mediapipe as mp
import numpy as np
import cv2 as ocv
from bounding_box import bounding_box as bb

# %%
# Tools
class recognizer:
    def __init__(self, modelPath: string, numHands: int = 2):
        """
        This method is used to initialize mediapipe recognizer

        Arguments
        =========
        modelPath: Absolute path of model file
        numHands: Number of hands to recognize from the image stream
        """
        self.modelPath = modelPath
        self.numHands = numHands
        self.opts = mp.tasks.vision.GestureRecognizerOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path=self.modelPath),
            running_mode = mp.tasks.vision.RunningMode.IMAGE,
            num_hands = self.numHands
        )
        self.recog = mp.tasks.vision.GestureRecognizer.create_from_options(self.opts)
        # self.handKeys = {
        #     "Closed_Fist": 0,
        #     "Open_Palm": 1,
        #     "Victory": 2,
        #     "Thumb_Up": 3,
        #     "Thumb_Down": 4,
        #     "Pointing_Up": 5
        # }
        self.handKeys = {
            "None": "None",
            "Closed_Fist": "Disarm",
            "Open_Palm": "Back",
            "Victory": "Forward",
            "Thumb_Up": "Right",
            "Thumb_Down": "Left",
            "Pointing_Up": "Arm", 
            "ILoveYou": "XXX"
        }
    
    def plot(self, img: np.array, instances: dict):
        """
        This method is used to plot the bboxes and landmarks on image

        Arguments
        =========
        img : OpenCV BGR image
        instances : Inferred instances

        Outputs
        =======
        Plotted image
        """
        height, width, channel = img.shape
        for instance in instances:
            bb.add(
                img,
                int(instance['bbox'][0] * width),
                int(instance['bbox'][1] * height),
                int(instance['bbox'][2] * width),
                int(instance['bbox'][3] * height),
                instance['gestureClass'],
                'red'
            )
            for coord in instance['lmks']:
                img = ocv.circle(img, (int(coord[0] * width), int(coord[1] * height)), 2, (0, 0, 255), 2) 
        return img

    def __parser__(self, _paRes):
        """
        This method is used to parse recognizer inference
        """
        hClass = _paRes.handedness
        hGuest = _paRes.gestures
        halmks = _paRes.hand_landmarks
        fdat = list()
        for i in range(len(hClass)):
            ldat = {
                'handClass': hClass[i][0].category_name,
                'handScore': hClass[i][0].score,
                'gestureClass': self.handKeys[hGuest[i][0].category_name],
                'gestureScore': hGuest[i][0].score,
                'lmks': list()
            }
            for j in halmks[i]:
                ldat['lmks'].append([j.x, j.y])
            ldat['lmks'] = np.array(ldat['lmks'])
            ldat['bbox'] = [
                np.amin(ldat['lmks'][:, 0]),
                np.amin(ldat['lmks'][:, 1]),
                np.amax(ldat['lmks'][:, 0]),
                np.amax(ldat['lmks'][:, 1])
            ]
            fdat.append(ldat)
        return fdat

    def __call__(self, img: np.array):
        """
        This method is used to perform inference on the suject image for hand recognition

        Arguments
        =========
        img : OpenCV image as BGR channel format

        Outputs
        =======
        Parsed inference from mediapipe hand guesture recogntiion
        """
        img = ocv.cvtColor(img, ocv.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB , data=img)
        return self.__parser__(self.recog.recognize(img))

# %%
# Execution
if __name__=="__main__":
    print("PLEASE IMPORT FILE TO CONTINUE")