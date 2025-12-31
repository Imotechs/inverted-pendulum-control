import mujoco
import mujoco.viewer
import numpy as np
import time

# ------------------ LOAD MODEL ------------------
model_path = "xml/mobile_pendulum.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# --------------- CONTROLLER GAINS -----------------
Kp_bal = 40.0       # a bit smaller to start
Kd_bal = 6.0

# Always-active controller (remove dead threshold so we react early)
USE_UPRIGHT_THRESHOLD = False
UPRIGHT_THRESHOLD = 0.0

# actuator indices (matches xml order)
LEFT_CTRL_IDX = 0
RIGHT_CTRL_IDX = 1

# safety saturation (keep small so wheels don't blast away)
MAX_VEL = 200.0       # target wheel velocity [rad/s]

# ---------------- DISTURBANCE CONFIG ----------------
# mode: "impulse_qvel" (instant angular velocity kick),
#       "impulse_torque" (short small torque), or
#       "noise" (small continuous torque noise)
DISTURBANCE_MODE = "impulse_qvel"   # "impulse_qvel", "impulse_torque", or "noise"

# impulse (qvel) parameters: instantaneous angular velocity added (rad/s)
QVEL_KICK_MAG = 0.35     # try 0.2..0.6 for gentle pushes
IMPULSE_INTERVAL = 3.0   # average seconds between kicks

# impulse (torque) parameters: if using torque-style impulses
IMPULSE_DURATION = 0.02  # seconds (very short)
IMPULSE_AMPLITUDE = 1.0  # N·m (small)

# noise parameters (if using noise)
NOISE_STD = 0.05         # small continuous torque noise (N·m)

# random seed (for reproducibility)
rng = np.random.default_rng(12345)

# ----------------- indices --------------------
pend_qpos_idx = model.joint("pendulum_hinge").qposadr[0]
pend_qvel_idx = model.joint("pendulum_hinge").dofadr[0]

# runtime state
next_impulse_time = time.time() + rng.uniform(0.3, IMPULSE_INTERVAL)
impulse_time_remaining = 0.0

# optional simple velocity filter for derivative
filtered_pend_vel = 0.0
vel_filter_alpha = 0.2

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetData(model, data)

    # small initial tilt so the controller needs to act
    #data.qpos[pend_qpos_idx] = 0.08   # ~4.6 degrees - gentle
    mujoco.mj_forward(model, data)

    last = time.time()
    while viewer.is_running():
        now = time.time()
        dt = now - last if last is not None else model.opt.timestep
        last = now

        # read pendulum state
        pend_angle = data.qpos[pend_qpos_idx]
        pend_vel_raw = data.qvel[pend_qvel_idx]

        # wrap angle to [-pi, pi]
        pend_angle = (pend_angle + np.pi) % (2*np.pi) - np.pi

        # filtered velocity
        filtered_pend_vel = (1.0 - vel_filter_alpha) * filtered_pend_vel + vel_filter_alpha * pend_vel_raw
        pend_vel = filtered_pend_vel

        # ----- controller (PD on pendulum angle) -----
        if USE_UPRIGHT_THRESHOLD and abs(pend_angle) <= UPRIGHT_THRESHOLD:
            control = 0.0
        else:
            # map angular error -> wheel target velocity (small gains)
            control = -(Kp_bal * pend_angle + Kd_bal * pend_vel)

        # scale control down so it doesn't produce extreme wheel velocity targets
        # this keeps actuator torques moderate and avoids runaway movement
        control = np.clip(control, -MAX_VEL, MAX_VEL)

        data.ctrl[LEFT_CTRL_IDX]  = control
        data.ctrl[RIGHT_CTRL_IDX] = control

        # ----- disturbances -----
        # clear previous applied generalized forces/vel kicks only on the hinge dof
        data.qfrc_applied[pend_qvel_idx] = 0.0

        if DISTURBANCE_MODE == "noise":
            # tiny continuous torque noise
            torque = rng.normal(0.0, NOISE_STD)
            data.qfrc_applied[pend_qvel_idx] += torque

        elif DISTURBANCE_MODE == "impulse_torque":
            # short small torque pulses (gentle)
            tnow = time.time()
            if tnow >= next_impulse_time:
                impulse_time_remaining = IMPULSE_DURATION
                next_impulse_time = tnow + rng.uniform(IMPULSE_INTERVAL * 0.5, IMPULSE_INTERVAL * 1.5)
                print(f"[disturb] starting small torque impulse (amp={IMPULSE_AMPLITUDE:.2f} N·m)")

            if impulse_time_remaining > 0.0:
                sign = rng.choice([-1.0, 1.0])
                data.qfrc_applied[pend_qvel_idx] += sign * IMPULSE_AMPLITUDE
                impulse_time_remaining -= dt

        elif DISTURBANCE_MODE == "impulse_qvel":
            # instantaneous angular-velocity "kick" (gentle). This is less likely to overpower a PD velocity-driven controller.
            tnow = time.time()
            if tnow >= next_impulse_time:
                # apply a one-shot velocity increment
                sign = rng.choice([-1.0, 1.0])
                delta = sign * QVEL_KICK_MAG
                data.qvel[pend_qvel_idx] += delta
                print(f"[disturb] qvel kick: delta={delta:.3f} rad/s")
                # schedule next
                next_impulse_time = tnow + rng.uniform(IMPULSE_INTERVAL * 0.5, IMPULSE_INTERVAL * 1.5)

        # ----- step simulator & render -----
        mujoco.mj_step(model, data)
        viewer.sync()