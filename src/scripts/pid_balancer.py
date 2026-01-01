# import mujoco
# import mujoco.viewer
# import numpy as np
# import time
# from collections import deque

# MODEL_PATH = "xml/mobile_pendulum.xml"
# model = mujoco.MjModel.from_xml_path(MODEL_PATH)
# data = mujoco.MjData(model)

# # ========== CONTROL PARAMETERS ==========
# # PID gains (tuned for realistic model)
# Kp = 5.0     # Proportional gain
# Ki = 0.5      # Integral gain (careful with windup)
# Kd = 2.0     # Derivative gain

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     mujoco.mj_resetData(model, data)
#     mujoco.mj_forward(model, data)

#     last = time.time()
#     step = 0
#     pendulum_joint = model.joint("pendulum_hinge")
#     pend_qpos_idx = pendulum_joint.qposadr[0]
#     pend_qvel_idx = pendulum_joint.dofadr[0]
#     # Recovery tracking
#     recovery_start_time = None
    
#     while viewer.is_running():
#         now = time.time()
#         dt = now - last
#         last = now

#         # ========== READ SENSOR DATA ==========
#         pend_angle = data.qpos[pend_qpos_idx]
#         pend_vel = data.qvel[pend_qvel_idx]
#         mujoco.mj_step(model, data)
#         viewer.sync()


import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "xml/slider.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ===================== PID GAINS =====================
Kp = 33.0
Ki = 5.0
Kd = 5.0

# integral term storage
integral_error = 0.0

# anti-windup limits
INTEGRAL_LIMIT = 2.0

# wheel parameters (must match XML)
wheel_radius = 0.10
wheel_distance = 0.34       # distance between wheels

# indices
pendulum_joint = model.joint("pendulum_hinge")
pend_qpos_idx = pendulum_joint.qposadr[0]
pend_qvel_idx = pendulum_joint.dofadr[0]

left_motor_id = model.actuator("left_motor").id
right_motor_id = model.actuator("right_motor").id
DEADZONE = 0.003  # radians ≈ 0.057 degrees
with mujoco.viewer.launch_passive(model, data) as viewer:

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    last_time = time.time()
    counter = 0
    while viewer.is_running():

        now = time.time()
        dt = now - last_time
        last_time = now

        # ===================== READ PENDULUM STATE =====================
        theta = data.qpos[pend_qpos_idx]          # pendulum angle (rad)
        theta_dot = data.qvel[pend_qvel_idx]      # angular velocity
        # normalize angle to [-pi, pi]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        print("pend. angle:",theta)
        data.qpos[pend_qpos_idx] = 0.03
        # ===================== PID CONTROLLER =====================
        error = 0.0 - theta
        #print("error:",error)
        # dead-zone: don't correct tiny angles
        if counter>800 and abs(theta) < DEADZONE:
            error = 0.0
            integral_error = 0.0
            #theta_dot = 0.0

        # proportional
        P = Kp * error

        # integral + anti-windup
        integral_error += error * dt
        integral_error = np.clip(integral_error, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        I = Ki * integral_error

        # derivative
        D = Kd * (-theta_dot)

        # cart velocity command
        cart_velocity_cmd = P + I + D

        # ===================== MAP CART VELOCITY TO WHEELS =====================
        # differential drive
        left_wheel_vel = cart_velocity_cmd / wheel_radius
        right_wheel_vel = cart_velocity_cmd / wheel_radius

        # apply command to actuators
        data.ctrl[left_motor_id] = left_wheel_vel
        data.ctrl[right_motor_id] = right_wheel_vel

        # ===================== STEP SIMULATION =====================
        mujoco.mj_step(model, data)

        # update viewer
        viewer.sync()
