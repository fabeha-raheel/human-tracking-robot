from HumanTracker import HumanTracker
import rospy

tracking_ugv = HumanTracker(auto_init=True)
tracking_ugv.start_human_detection()
# tracking_ugv.motor_control_target()
tracking_ugv.start_tracking()

while True:
    if rospy.is_shutdown():
        print("Shutting down...")
        tracking_ugv.kill = True
        break
    
print("Program End.")

