import cv2
import threading

class HumanTracker():

    def __init__(self) -> None:
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,1280)
        self.cap.set(4,720)
        self.cap.set(10,70)

        self.thres = 0.45 # Threshold to detect object

        self.classNames= []

        self.classFile = 'coco.names'
        self.configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        self.weightsPath = 'frozen_inference_graph.pb'

        self.kill = False

    def init_detector(self):

        with open(self.classFile, 'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

        self.net = cv2.dnn_DetectionModel(self.weightsPath,self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/ 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

    def start_human_detection(self):

        self.detection_thread = threading.Thread(target=self.human_detection_target, daemon=True) 

    def human_detection_target(self):
        
        while not self.kill:
            success,img = self.cap.read()
            classIds, confs, bbox = self.net.detect(img,confThreshold=self.thres)
            # print(classIds,bbox)

            if len(classIds) != 0:
                for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                    if classId == 1:
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                        cv2.putText(img,self.classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

            cv2.imshow("Output",img)
            cv2.waitKey(1)

