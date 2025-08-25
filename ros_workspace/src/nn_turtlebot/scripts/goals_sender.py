#!/usr/bin/env python3

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
import math
import random
import signal

def signal_handler(sig, frame):
    """Handle Ctrl+C interruption by shutting down the node."""
    rospy.signal_shutdown("User interrupted")

def euler_to_quaternion(yaw):
    """Manually convert yaw (theta) to quaternion for rotation around z-axis."""
    roll = 0
    pitch = 0
    yaw = yaw

    # Calculate half angles
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Quaternion calculation
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return Quaternion(x, y, z, w)

def send_goal(x, y, theta):
    """Send a navigation goal to the move_base action server."""
    goal_msg = MoveBaseGoal()
    goal_msg.target_pose.header.frame_id = "map"
    goal_msg.target_pose.header.stamp = rospy.Time.now()
    goal_msg.target_pose.pose.position.x = x
    goal_msg.target_pose.pose.position.y = y
    # Convert theta to quaternion manually
    quat = euler_to_quaternion(theta)
    goal_msg.target_pose.pose.orientation = quat
    
    client.send_goal(goal_msg)
    result = client.wait_for_result(rospy.Duration(150))  # Wait to reach the goal
    if result:
        if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal reached: ({}, {}, {})".format(x, y, theta))
            rospy.sleep(1.5)  # Sleep for 1.5 sec before sending the next goal
        else:
            rospy.logwarn("Goal failed: ({}, {}, {})".format(x, y, theta))
    else:
        rospy.logwarn("Timeout waiting for goal: ({}, {}, {})".format(x, y, theta))

if __name__ == '__main__':
    rospy.init_node('goal_sender')
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()
    
    # List of goals with theta = 0
    goals = [
        (-0.76, 2.01, 0.0),
        (-2.71, 3.77, 0.0),
        (-4.88, 3.09, 0.0),
        (-6.36, 1.65, 0.0),
        (0.67, 0.31, 0.0),
        (-2.84, 0.97, 0.0),
        (2.76, 0.54, 0.0),
        (4.38, 1.72, 0.0),
        (4.81, 3.26, 0.0),
        (6.83, 3.26, 0.0),
        (4.52, 4.20, 0.0),
        (1.28, 3.55, 0.0),
        (0.55, 2.11, 0.0),
        (6.04, 0.03, 0.0),
        (3.27, 0.93, 0.0),
        (6.67, -1.24, 0.0),
        (-4.42, 2.43, 0.0),
        (-6.37, 4.14, 0.0),
        (-6.05, -1.73, 0.0),
        (-2.56, 3.48, 0.0),
        (0.94, 3.53, 0.0),
        (1.12, 1.48, 0.0),
        (5.77, 2.83, 0.0),
        (5.67, 4.23, 0.0),
        (6.65, 0.17, 0.0),
        (5.35, -2.36, 0.0),
        (3.94, 1.52, 0.0),
        (-6.79, 4.50, 0.0),
        (6.80, 4.52, 0.0),
        (6.45, -1.02, 0.0),
        (6.54, 3.96, 0.0),
        (-5.98, 3.93, 0.0),
        (-6.13, -0.66, 0.0),
        (-1.45, 4.13, 0.0),
        (-6.54, -3.02, 0.0),
        (1.05, 2.99, 0.0),
        (5.61, 1.17, 0.0),
        (5.77, -4.16, 0.0),
        (-2.06, 4.14, 0.0),
        (1.03, -0.90, 0.0),
        (3.07, 4.21, 0.0),
        (-6.17, 4.14, 0.0),
        (-6.48, -1.97, 0.0)
    ]
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        first_run = True
        while not rospy.is_shutdown():
            if first_run:
                for goal in goals:
                    send_goal(*goal)
                first_run = False
            else:
                # Randomly select a goal from the list
                goal = random.choice(goals)
                send_goal(*goal)
    except Exception as e:
        rospy.logerr("An error occurred: {}".format(e))
    
    rospy.loginfo("Data collection stopped")