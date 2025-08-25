#!/usr/bin/env python3
"""
nn_controller.py  — Hybrid NN + Mode-0 Sequential PID
──────────────────────────────────────────────────────
• Far from the goal (> DIST_TOL):  commands come from the trained
  neural network stored in MODEL_FILE.

• Near the goal (≤ DIST_TOL):  the node switches into
  "Mode 0 / Sequential PID":
      1. Align robot heading to the goal.
      2. Drive straight toward the goal.
      3. Final heading align to θ_r (here fixed to 0).
  The three phases run one after another, using the *same* PID gains
  tuned in Lab 3.

• When all sub-phases finish, the node stops the robot and publishes
  /goal_reached = True so motion_planner can prompt for the next goal.

Only standard ROS msgs are used:
  ─ subscribes:
      /gazebo/model_states        (gazebo_msgs/ModelStates)
      /reference_pose             (std_msgs/Float64MultiArray) – [x_r, y_r]
  ─ publishes:
      /cmd_vel                    (geometry_msgs/Twist)
      /goal_reached               (std_msgs/Bool)
"""

# ──────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────
import rospy, math, h5py, numpy as np
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Bool

# ──────────────────────────────────────────────────────────
# 0. Configurable constants
# ──────────────────────────────────────────────────────────
ROBOT_NAME      = "turtlebot3_waffle_pi"        # Gazebo model.name
MODEL_FILE      = "/home/ubuntu/lab8_lan_pham_ws/model.h5"

MAX_LINEAR_VEL  = 0.3      # [m/s]
MAX_ANGULAR_VEL = 1.9       # [rad/s]

DIST_TOL        = 0.20      # [m]  distance that triggers PID hand-over

# PID gains (identical to your Lab-3 Mode-0 controller)
Kp_lin, Ki_lin, Kd_lin = 0.4, 0.0001, 0.0001
Kp_ang, Ki_ang, Kd_ang = 1.0, 0.0001, 0.0001

# ──────────────────────────────────────────────────────────
# 1. Global state containers
# ──────────────────────────────────────────────────────────
current_state = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
goal_state    = {'x': 0.0, 'y': 0.0}      # goal θ is fixed to 0
goal_reached_flag = False                   # published once per goal

# NN weights will be filled with [(W1,b1), (W2,b2), …]
nn_weights = []

# PID phase state machine
pid_state = {
    "phase"      : "align1",   # align1 → drive → align2
    "int_lin"    : 0.0,
    "int_ang"    : 0.0,
    "prev_lin"   : 0.0,
    "prev_ang"   : 0.0,
    "last_time"  : None,       # for Δt calculation
}

# ──────────────────────────────────────────────────────────
# 2. Helper functions
# ──────────────────────────────────────────────────────────
def wrap_to_pi(a):
    """Map any angle to (-π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def clamp(x, lo, hi):
    return max(min(x, hi), lo)

# ──────────────────────────────────────────────────────────
# 3. H5 weight loader (file-structure-agnostic)
# ──────────────────────────────────────────────────────────
def collect_dense(group, out):
    """Recursively append any subgroup with datasets 'kernel' & 'bias'."""
    for k in group:
        g = group[k]
        if isinstance(g, h5py.Group):
            if 'kernel' in g and 'bias' in g:
                out.append((np.array(g['kernel']), np.array(g['bias'])))
            else:
                collect_dense(g, out)

def load_model_weights():
    global nn_weights
    with h5py.File(MODEL_FILE, 'r') as f:
        if 'model_weights' not in f:
            rospy.logerr("model_weights group missing in H5.")
            return False
        nn_weights.clear()
        collect_dense(f['model_weights'], nn_weights)

    if not nn_weights:
        rospy.logerr("No Dense layers found in model.")
        return False

    rospy.loginfo(f"Loaded {len(nn_weights)} Dense layers.")
    for i, (W, b) in enumerate(nn_weights, 1):
        rospy.loginfo(f"  L{i}: W{W.shape}, b{b.shape}")
    return True

# ──────────────────────────────────────────────────────────
# 4. Pure-NumPy forward pass
# ──────────────────────────────────────────────────────────
def nn_forward(vec):
    a = vec
    for i, (W, b) in enumerate(nn_weights):
        z = a @ W + b
        a = np.maximum(z, 0.0) if i < len(nn_weights)-1 else z
    return a                     # → [v_pred, w_pred]

# ──────────────────────────────────────────────────────────
# 5. PID Mode-0 Sequential controller
# ──────────────────────────────────────────────────────────
def pid_sequential(x, y, th, gx, gy, s):
    """
    Executes one iteration of Mode-0 sequential PID.
    Returns (v_cmd, w_cmd, finished_bool).
    """
    # Δt for derivative / integral
    now   = rospy.Time.now().to_sec()
    dt    = now - s["last_time"] if s["last_time"] else 0.1
    s["last_time"] = now

    # Errors
    dx, dy = gx - x, gy - y
    dist   = math.hypot(dx, dy)
    angle_to_target = math.atan2(dy, dx)
    err_heading     = wrap_to_pi(angle_to_target - th)
    err_final_theta = wrap_to_pi(0.0 - th)        # desired θ = 0

    # Phase thresholds
    HEADING_OK  = 0.10   # [rad]
    POSITION_OK = 0.10   # [m]
    ANGLE_OK    = 0.10   # [rad]

    phase = s["phase"]

    if phase == "align1":
        lin_err, ang_err = 0.0, err_heading
        if abs(err_heading) < HEADING_OK:
            rospy.loginfo("PID phase → drive")
            s["phase"] = "drive"

    elif phase == "drive":
        lin_err, ang_err = dist, 0.0
        if dist < POSITION_OK:
            rospy.loginfo("PID phase → align2")
            s["phase"] = "align2"

    else:  # align2
        lin_err, ang_err = 0.0, err_final_theta
        if abs(err_final_theta) < ANGLE_OK:
            return 0.0, 0.0, True   # done!

    # ---- Linear PID -------------------------------------------------
    lin_P = Kp_lin * lin_err
    s["int_lin"] += lin_err * dt
    lin_I = Ki_lin * s["int_lin"]
    lin_D = Kd_lin * ((lin_err - s["prev_lin"]) / dt)
    s["prev_lin"] = lin_err
    v_cmd = lin_P + lin_I + lin_D

    # ---- Angular PID ------------------------------------------------
    ang_P = Kp_ang * ang_err
    s["int_ang"] += ang_err * dt
    ang_I = Ki_ang * s["int_ang"]
    ang_D = Kd_ang * ((ang_err - s["prev_ang"]) / dt)
    s["prev_ang"] = ang_err
    w_cmd = ang_P + ang_I + ang_D

    # Clamp to safe limits
    return clamp(v_cmd, -MAX_LINEAR_VEL, MAX_LINEAR_VEL), \
           clamp(w_cmd, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL), \
           False

# ──────────────────────────────────────────────────────────
# 6. ROS callbacks
# ──────────────────────────────────────────────────────────
def model_states_cb(msg):
    try:
        idx = msg.name.index(ROBOT_NAME)
    except ValueError:   # our robot not in list
        return
    pose = msg.pose[idx]
    x = pose.position.x
    y = pose.position.y
    q = pose.orientation
    siny = 2*(q.w*q.z + q.x*q.y)
    cosy = 1 - 2*(q.y*q.y + q.z*q.z)
    yaw  = math.atan2(siny, cosy)

    current_state['x'] = x
    current_state['y'] = y
    current_state['theta'] = yaw

def reference_pose_cb(msg):
    global goal_reached_flag
    if len(msg.data) < 2:
        rospy.logwarn("reference_pose expects [x,y]")
        return
    goal_state['x'] = float(msg.data[0])
    goal_state['y'] = float(msg.data[1])
    goal_reached_flag = False
    # reset PID phase machine
    pid_state.update({"phase":"align1", "int_lin":0.0, "int_ang":0.0,
                      "prev_lin":0.0, "prev_ang":0.0, "last_time":None})
    rospy.loginfo(f"New goal set → ({goal_state['x']:.2f}, {goal_state['y']:.2f})")

# ──────────────────────────────────────────────────────────
# 7. Main
# ──────────────────────────────────────────────────────────
def main():
    global goal_reached_flag

    rospy.init_node("nn_controller", anonymous=False)

    if not load_model_weights():
        return

    # Publishers / subscribers
    cmd_pub          = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    goal_flag_pub    = rospy.Publisher("/goal_reached", Bool,  queue_size=1)
    rospy.Subscriber("/gazebo/model_states",  ModelStates,           model_states_cb)
    rospy.Subscriber("/reference_pose",       Float64MultiArray,     reference_pose_cb)

    rate = rospy.Rate(10)   # Hz
    rospy.loginfo("Hybrid NN + PID controller running…")

    while not rospy.is_shutdown():

        # ensure we have everything
        if None in current_state.values() or None in goal_state.values():
            rate.sleep()
            continue

        x,y,th = current_state['x'], current_state['y'], current_state['theta']
        gx,gy  = goal_state['x'],      goal_state['y']

        dist   = math.hypot(gx-x, gy-y)

        # ------------------------------------------------------------------
        #  A) FAR  →  use NN
        # ------------------------------------------------------------------
        if dist >= DIST_TOL:
            input_vec = np.array([x, y, th, gx, gy, 0.0], dtype=np.float32)
            v_pred, w_pred = nn_forward(input_vec)
            v_cmd = clamp(v_pred, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
            w_cmd = clamp(w_pred, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

            print(f"[NN] dist={dist:.2f}  v={v_cmd:.3f}  w={w_cmd:.3f}")


        # ------------------------------------------------------------------
        #  B) NEAR →  switch to sequential PID
        # ------------------------------------------------------------------
        else:
            v_cmd, w_cmd, finished = pid_sequential(x, y, th, gx, gy, pid_state)
            print(f"[PID-{pid_state['phase']}] v={v_cmd:.3f}  w={w_cmd:.3f}")

            if finished and not goal_reached_flag:
                rospy.loginfo("PID finished → goal reached!")
                goal_flag_pub.publish(Bool(data=True))
                goal_reached_flag = True

        # Always publish Twist (even zeros after finish so robot stays still)
        twist = Twist()
        twist.linear.x  = v_cmd
        twist.angular.z = w_cmd
        cmd_pub.publish(twist)

        rate.sleep()

    # ─ graceful shutdown ─
    cmd_pub.publish(Twist())   # stop
    rospy.sleep(1.0)

# entry point
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
