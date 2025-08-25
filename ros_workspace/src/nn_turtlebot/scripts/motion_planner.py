#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray, Bool

# Global flag for goal_reached
goal_reached = False

def goal_reached_callback(msg):
    global goal_reached
    goal_reached = msg.data

if __name__ == "__main__":
    rospy.init_node("motion_planner", anonymous=True)

    # Publisher for /reference_pose as Float64MultiArray
    ref_pub = rospy.Publisher("/reference_pose", Float64MultiArray, queue_size=1)
    # Subscriber for /goal_reached
    rospy.Subscriber("/goal_reached", Bool, goal_reached_callback)

    rospy.sleep(1.0)  # let ROS connections settle
    rospy.loginfo("motion_planner started. Enter goals (x y).")

    while not rospy.is_shutdown():
        # 1) Get (x, y) from user input
        try:
            x_r = float(input("Enter target X: "))
            y_r = float(input("Enter target Y: "))
        except ValueError:
            print("Invalid input; enter numeric X and Y.")
            continue

        # 2) Publish exactly the input data in a Float64MultiArray
        target_msg = Float64MultiArray()
        target_msg.data = [x_r, y_r]  # only x, y (theta is default 0)
        goal_reached = False
        ref_pub.publish(target_msg)
        rospy.loginfo(f"Published [x={x_r:.2f}, y={y_r:.2f}] → waiting for /goal_reached...")

        # 3) Wait until nn_controller sets goal_reached = True
        while not rospy.is_shutdown() and not goal_reached:
            rospy.sleep(0.1)

        if rospy.is_shutdown():
            break

        rospy.loginfo("Goal reached!")

    rospy.loginfo("motion_planner shutting down.")
