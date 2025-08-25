import rosbag
import pandas as pd
from tf.transformations import euler_from_quaternion
import numpy as np

# Configuration
bag_file = "Recorded_Robot.bag"
robot_name = "turtlebot3_waffle_pi"  # Adjust if needed

# Storage
dataset = []
current_pose = None
goal_pose = None

def get_yaw(quat):
    orientation_list = [quat.x, quat.y, quat.z, quat.w]
    _, _, yaw = euler_from_quaternion(orientation_list)
    return yaw

# Process the bag file
with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states', '/cmd_vel', '/move_base/goal']):
        if topic == '/gazebo/model_states':
            try:
                index = msg.name.index(robot_name)
                pose = msg.pose[index]
                x = pose.position.x
                y = pose.position.y
                theta = get_yaw(pose.orientation)
                current_pose = [x, y, theta]
            except ValueError:
                pass  # Robot not found
        elif topic == '/move_base/goal':
            goal = msg.goal.target_pose.pose
            goal_x = goal.position.x
            goal_y = goal.position.y
            goal_theta = get_yaw(goal.orientation)
            goal_pose = [goal_x, goal_y, goal_theta]
        elif topic == '/cmd_vel':
            if current_pose is not None and goal_pose is not None:
                v = msg.linear.x
                w = msg.angular.z
                data_point = current_pose + goal_pose + [v, w]
                dataset.append(data_point)

# Save to CSV
columns = ['x', 'y', 'theta', 'goal_x', 'goal_y', 'goal_theta', 'v', 'w']
df = pd.DataFrame(dataset, columns=columns)
df.to_csv('training_data.csv', index=False)
print("Saved dataset to training_data.csv")