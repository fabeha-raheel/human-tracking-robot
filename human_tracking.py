from HumanTracker import HumanTracker

tracking_ugv = HumanTracker(auto_init=True)
tracking_ugv.start_human_detection()
tracking_ugv.motor_control_target()