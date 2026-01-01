# import mujoco
# import mujoco.viewer
# import numpy as np
# import time
# import random
# MODEL_PATH = "xml/slider.xml"

# model = mujoco.MjModel.from_xml_path(MODEL_PATH)
# data = mujoco.MjData(model)

# # ------------ PID GAINS -------------
# Kp = 120.0
# Ki = 15.0
# Kd = 50.0

# integral_err = 0.0
# prev_err = 0.0

# target_angle = 0.0  # upright
# DEADZONE = 0.003
# pend_idx = model.joint("pendulum_hinge").qposadr[0]
# motor_id = model.actuator("cart_motor").id

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     mujoco.mj_resetData(model, data)

#     last = time.time()
#     data.qpos[pend_idx] = random.choice([-0.01, 0.01])
#     counter = 0

#     while viewer.is_running():

#         dt = model.opt.timestep

#         # -------- read pendulum state --------
#         angle = data.qpos[pend_idx]
#         if counter>8000 and abs(angle) < 0.0001:
#             #data.qpos[pend_idx] = random.choice([-0.01, 0.01])
#             data.qvel[pend_idx] = random.choice([-0.2, 0.1, -0.1, 0.2, 0.3, -0.3])
#             counter = 0
#             prev_err = 0.0
#             integral_err = 0.0

#         else:
#             counter += 1
#         # wrap angle to (-pi, pi)
#         angle = ((angle + np.pi) % (2*np.pi)) - np.pi

#         vel = data.qvel[pend_idx]
#         print("Angle:",angle)
#         # PID error
#         err = target_angle - angle

#         integral_err += err * dt
#         integral_err = np.clip(integral_err, -2.0, 2.0)

#         derr = (err - prev_err) / dt
#         prev_err = err

#         # PID output
#         u = Kp * err + Ki * integral_err + Kd * derr

#         # VERY IMPORTANT: direction of correction
#         u = -u          # move cart TOWARD fall

#         u = np.clip(u, -20.0, 20.0)
#         if abs(angle)<=DEADZONE:
#             u = -u
#         data.ctrl[motor_id] = u

#         # now step physics
#         mujoco.mj_step(model, data)

#         viewer.sync()

import mujoco
import mujoco.viewer
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import signal
import sys

MODEL_PATH = "xml/slider.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ------------ PID GAINS -------------
Kp = 120.0
Ki = 15.0
Kd = 50.0

integral_err = 0.0
prev_err = 0.0

target_angle = 0.0  # upright
DEADZONE = 0.003
pend_idx = model.joint("pendulum_hinge").qposadr[0]
motor_id = model.actuator("cart_motor").id

# ------------ Data recording -------------
times = []
angles = []
velocities = []
controls = []

# handle Ctrl+C to exit cleanly
stop_simulation = False
def signal_handler(sig, frame):
    global stop_simulation
    stop_simulation = True
signal.signal(signal.SIGINT, signal_handler)

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetData(model, data)

    last = time.time()
    data.qpos[pend_idx] = random.choice([-0.01, 0.01])
    counter = 0

    while viewer.is_running() and not stop_simulation:

        dt = model.opt.timestep
        now = time.time()

        # -------- read pendulum state --------
        angle = data.qpos[pend_idx]
        angles.append(angle)

        if counter>8000 and abs(angle) < 0.0001:
            data.qvel[pend_idx] = random.choice([-0.2, 0.1, -0.1, 0.2, 0.3, -0.3])
            counter = 0
            prev_err = 0.0
            integral_err = 0.0
        else:
            counter += 1

        # wrap angle to (-pi, pi)
        angle = ((angle + np.pi) % (2*np.pi)) - np.pi
        vel = data.qvel[pend_idx]
        velocities.append(vel)
        print("Angle:",angle)

        # -------- PID error --------
        err = target_angle - angle
        integral_err += err * dt
        integral_err = np.clip(integral_err, -2.0, 2.0)
        derr = (err - prev_err) / dt
        prev_err = err

        # PID output
        u = Kp * err + Ki * integral_err + Kd * derr
        u = -u  # move cart TOWARD fall
        u = np.clip(u, -20.0, 20.0)
        controls.append(u)
        # deadzone
        if abs(angle) <= DEADZONE:
            u = -u

        data.ctrl[motor_id] = u

        # -------- record data --------
        times.append(now - last)

        # step physics
        mujoco.mj_step(model, data)
        viewer.sync()

# ------------------ Plot results ------------------
plt.figure(figsize=(12,6))

plt.subplot(3,1,1)
plt.plot(np.cumsum(times), angles, label="Pendulum Angle (rad)")
plt.axhline(DEADZONE, color='r', linestyle='--', alpha=0.5)
plt.axhline(-DEADZONE, color='r', linestyle='--', alpha=0.5)
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(np.cumsum(times), velocities, label="Pendulum Angular Velocity (rad/s)")
plt.ylabel("Velocity")
plt.legend()
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(np.cumsum(times), controls, label="Cart Control Input")
plt.xlabel("Time (s)")
plt.ylabel("Control (u)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

