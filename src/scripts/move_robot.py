import mujoco
import mujoco.viewer
import numpy as np
import time

# ---------- Load your model ----------
model = mujoco.MjModel.from_xml_path("xml/mobile_pendulum.xml")
data = mujoco.MjData(model)

# ---------- OPTIONAL: give pendulum a tiny initial tilt ----------
pendulum_hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_hinge")
data.qpos[pendulum_hinge_id] = np.deg2rad(3)   # 3 degree tilt

# ---------- Simulation loop ----------
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    start = time.time()

    while viewer.is_running():
        step_start = time.time()

        t = data.time

        # ----------- SIMPLE OPEN LOOP DRIVE -----------
        # forward motion
        v = 5.0 * np.sin(0.3 * t)   # oscillating wheel speed

        # set wheel velocities (actuator order)
        data.ctrl[model.actuator("left_motor").id] = v
        data.ctrl[model.actuator("right_motor").id] = v

        # ----------- STEP PHYSICS -----------
        mujoco.mj_step(model, data)

        # slow to realtime
        viewer.sync()

        # ensure realtime pacing
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
