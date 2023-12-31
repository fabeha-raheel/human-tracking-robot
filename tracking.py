import cv2
import threading
import numpy as np
import math
from cvFunctions import *

class HumanTracker():

    def __init__(self, auto_init=True) -> None:
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,1280)
        self.cap.set(4,720)
        self.cap.set(10,70)

        self.thres = 0.6 # Threshold to detect object
        self.nms_threshold = 0.2

        self.configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        self.weightsPath = 'frozen_inference_graph.pb'

        self.classNames= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 
             'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 
             'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
             'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 
             'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
             'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
             'hair drier', 'toothbrush', 'hair brush']

        self.kill = False

        self.sp_x, self.sp_y = int(1280/2), int(720/2)

        if auto_init:
            self.init_detector()

    def init_detector(self):

        self.net = cv2.dnn_DetectionModel(self.weightsPath,self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/ 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def start_human_detection(self):

        self.detection_thread = threading.Thread(target=self.human_detection_target, daemon=True)
        self.detection_thread.start() 

    def start_tracking(self):
        self.control_thread = threading.Thread(target=self.motor_control_target, daemon=True)
        self.control_thread.start()

    def motor_control_target(self):
        pass

    def human_detection_target(self):
        
        while not self.kill:
            success,img = self.cap.read()
            classIds, confs, bbox = self.net.detect(img,confThreshold=self.thres)

            if len(classIds) != 0:
                bbox = list(bbox)
                confs = list(np.array(confs).reshape(1,-1)[0])
                confs = list(map(float,confs))

                indices = cv2.dnn.NMSBoxes(bbox,confs,self.thres,self.nms_threshold)
                indices = np.array([indices])
                
                try:
                    for i in indices:
                        if i[0] == 0:
                            i = i[0]
                            box = bbox[i]
                            x,y,w,h = box[0],box[1],box[2],box[3]
                            putTextRect(img, f'Human {math.ceil(confs[i-1]*100)}%', (max(0, x+8), max(35, y-13)), scale=1.5, thickness=2, colorR=(175,0,175))
                            cornerRect(img, (x, y, w, h), colorR=(175,0,175), rt=3, t=10, l=int(min(w, h)/8), colorC=(0,255,0))

                            self.cx, self.cy = int(x + (w//2)), int(y + (h//2))
                            cv2.line(img, (self.cx, self.cy), (self.sp_x, self.sp_y), (0,255,0), 3)
                            cv2.line(img, (self.cx, int(self.cy-h/2)+3), (self.cx, int(self.cy+h/2)-3), (0,0,0), 2)
                            cv2.line(img, (int(self.cx-w/2)+3, self.cy), (int(self.cx+w/2)-3, self.cy), (0,0,0), 2)
                            cv2.circle(img, (self.cx, self.cy), 5, (0, 0, 255), cv2.FILLED)
                            cv2.circle(img, (self.sp_x, self.sp_y), 7, (0, 0, 255), cv2.FILLED)
                except IndexError:
                    continue
                    
                    

            cv2.imshow("Output",img)
            cv2.waitKey(1)

