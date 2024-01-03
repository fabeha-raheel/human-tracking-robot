import cv2
import threading
import rospy

from handTester import *
from Rover import *

class GestureControlledUGV():
    def __init__(self):

        rospy.init_node('gesture_control', anonymous=True)
        rospy.on_shutdown(self.shutdown)

        self.kill = False
        self.action = None

        self.rover = Rover()
        self.gesture = recognizer(modelPath='/home/ugv/human-tracking-robot/gesture_recognizer.task')

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,1280)
        self.cap.set(4,720)
        self.cap.set(10,70)

    def start_gesture_recognition(self):
        self.gesture_recognition_thread = threading.Thread(target=self._gesture_recognition, daemon=True)
        self.gesture_recognition_thread.start()

    def start_ugv_control(self):
        self.motor_control_thread = threading.Thread(target=self._motor_control, daemon=True)
        self.motor_control_thread.start()

    def _gesture_recognition(self):

        while not self.kill:
            success,img = self.cap.read()
            out = self.gesture(img)
            # print(out)
            if out != []:
                self.action = out[0]['gestureClass']
                print('Gesture detected: ', self.action)

    def _motor_control(self):

        while not self.kill:

            if self.action == "Disarm":
                if self.rover.arm_status == True:
                    self.rover.disarm()
            elif self.action == "Arm":
                if self.rover.arm_status == False:
                    self.rover.arm()
            elif self.action == "Back":
                self.rover.move_backward()
            elif self.action == "Forward":
                self.rover.move_forward()
            elif self.action == "Right":
                self.rover.rotate_right()
            elif self.action == "Left":
                self.rover.rotate_left()
            else:
                self.rover.stop()

            time.sleep(0.2)

    def shutdown(self):
        self.rover.stop()
        if self.rover.arm_status == True:
            self.rover.disarm()
        time.sleep(0.2)
        self.kill = True


if __name__=="__main__":

    gc_ugv = GestureControlledUGV()
    gc_ugv.start_gesture_recognition()
    gc_ugv.start_ugv_control()
    rospy.spin()
