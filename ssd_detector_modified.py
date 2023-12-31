import cv2
import numpy as np
import math

from cvFunctions import *

thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,150)

classNames= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 
             'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 
             'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
             'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 
             'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
             'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
             'hair drier', 'toothbrush', 'hair brush']

# classFile = 'MobilenetSSD_Object_Detection/coco.names'
# with open(classFile,'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n')

# print(classNames)

configPath = 'MobilenetSSD_Object_Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'MobilenetSSD_Object_Detection/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

sp_x, sp_y = int(1280/2), int(720/2) 

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    indices = np.array([indices])
    # print(indices)
    

    for i in indices:
        if i[0] == 0:
            i = i[0]
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            putTextRect(img, f'Human {math.ceil(confs[i-1]*100)}%', (max(0, x+8), max(35, y-13)), scale=1.5, thickness=2, colorR=(175,0,175))
            cornerRect(img, (x, y, w, h), colorR=(175,0,175), rt=3, t=10, l=int(min(w, h)/8), colorC=(0,255,0))
            # img = cornerFilledRect(img, (x, y, w, h), colorR=(0,255,0), l=int(min(w, h)/8), colorC=(0,0,175))
            # cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
            # cv2.putText(img,classNames[classIds[i]-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            # putTextRect(img, f'{classNames[classIds[i]-1]} {math.ceil(confs[i-1]*100)}%', (max(0, x+8), max(35, y-13)), scale=1.5, thickness=2)

            cx, cy = int(x + (w//2)), int(y + (h//2))
            cv2.line(img, (cx, cy), (sp_x, sp_y), (0,255,0), 3)
            cv2.line(img, (cx, int(cy-h/2)+3), (cx, int(cy+h/2)-3), (0,0,0), 2)
            cv2.line(img, (int(cx-w/2)+3, cy), (int(cx+w/2)-3, cy), (0,0,0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (sp_x, sp_y), 7, (0, 0, 255), cv2.FILLED)
            
            

    cv2.imshow("Output",img)
    cv2.waitKey(1)