import mujoco
import mujoco.viewer
import numpy as np
import time
from collections import deque

MODEL_PATH = "xml/mobile_pendulum.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ========== CONTROL PARAMETERS ==========
# PID gains (tuned for realistic model)
Kp = 5.0     # Proportional gain
Ki = 0.5      # Integral gain (careful with windup)
Kd = 2.0     # Derivative gain

# Wheel damping (braking) - lower for faster response
WHEEL_DAMPING = 0.5

# Max wheel torque - increased for realistic model
MAX_WHEEL_TORQUE = 12.0

# Control sign (1.0 for normal, -1.0 if control is reversed)
CONTROL_SIGN = 1.0

# ========== STABILITY & DISTURBANCE PARAMETERS ==========
# Stability detection
STABILITY_ANGLE_THRESHOLD = 0.02      # rad (~1.1 degrees)
STABILITY_VEL_THRESHOLD = 0.08        # rad/s
STABLE_TIME_REQUIRED = 0.8           # seconds to be considered stable

# Disturbance parameters
DISTURBANCE_STRENGTH_MIN = 0.15       # rad/s minimum kick
DISTURBANCE_STRENGTH_MAX = 0.35       # rad/s maximum kick
DISTURBANCE_INTERVAL_MIN = 0.3        # min seconds between disturbances
DISTURBANCE_INTERVAL_MAX = 1.5        # max seconds between disturbances

# ========== CONTROL ACTIVATION THRESHOLDS ==========
CONTROL_ANGLE_THRESHOLD = 0.04        # rad - start control beyond this
CONTROL_VEL_THRESHOLD = 0.12          # rad/s - start control beyond this

# ========== FILTERING ==========
VELOCITY_FILTER_SIZE = 100              # Moving average filter for velocity

# ========== GET MODEL INDICES ==========
try:
    pendulum_joint = model.joint("pendulum_hinge")
    pend_qpos_idx = pendulum_joint.qposadr[0]
    pend_qvel_idx = pendulum_joint.dofadr[0]
except:
    print("Warning: 'pendulum_hinge' not found, trying default indices")
    pend_qpos_idx = 7  # Adjust based on your model
    pend_qvel_idx = 6  # Adjust based on your model

try:
    left_wheel_joint = model.joint("left_wheel_joint")
    right_wheel_joint = model.joint("right_wheel_joint")
    left_wheel_dof = left_wheel_joint.dofadr[0]
    right_wheel_dof = right_wheel_joint.dofadr[0]
except:
    print("Warning: Wheel joints not found, trying default indices")
    left_wheel_dof = 7  # Adjust based on your model
    right_wheel_dof = 8  # Adjust based on your model

# Get actuator indices if they exist
try:
    left_motor_idx = model.actuator("left_wheel_motor").id
    right_motor_idx = model.actuator("right_wheel_motor").id
    use_actuators = True
    print("Found motor actuators, using actuator control")
except:
    use_actuators = False
    print("Using direct joint control via ctrl array")
    left_motor_idx = 0
    right_motor_idx = 1

print(f"Indices - Pendulum: pos={pend_qpos_idx}, vel={pend_qvel_idx}")
print(f"          Wheels: left={left_wheel_dof}, right={right_wheel_dof}")

# ========== PID CONTROLLER CLASS ==========
class PIDController:
    def __init__(self, kp, ki, kd, max_output, dt, windup_limit=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.dt = dt
        self.windup_limit = windup_limit
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = 0.0
        self.vel_history = deque(maxlen=VELOCITY_FILTER_SIZE)
        
    def update(self, error, measurement, measurement_vel=None):
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * self.dt
        # Clamp integral to prevent windup
        max_integral = self.windup_limit / self.ki if self.ki > 0 else 0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        i_term = self.ki * self.integral
        
        # Derivative term (filtered)
        if measurement_vel is not None:
            velocity = measurement_vel
        else:
            velocity = (measurement - self.prev_measurement) / self.dt
        
        self.vel_history.append(velocity)
        filtered_vel = np.mean(self.vel_history) if self.vel_history else velocity
        d_term = self.kd * filtered_vel
        
        # Combine terms
        output = p_term + i_term - d_term  # Negative derivative for damping
        
        # Output saturation
        output = np.clip(output, -self.max_output, self.max_output)
        
        # Update state
        self.prev_error = error
        self.prev_measurement = measurement
        
        return output
    
    def reset_integral(self):
        self.integral = 0.0

# ========== INITIALIZATION ==========
pid_controller = PIDController(Kp, Ki, Kd, MAX_WHEEL_TORQUE, model.opt.timestep)

# Stability monitoring
stable_start_time = None
last_disturbance_time = time.time()
next_disturbance_time = time.time() + np.random.uniform(DISTURBANCE_INTERVAL_MIN, DISTURBANCE_INTERVAL_MAX)

# Performance tracking
disturbance_count = 0
recovery_times = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetData(model, data)
    
    # Small initial tilt (about 3.4 degrees)
    #data.qpos[pend_qpos_idx] = 0.06
    mujoco.mj_forward(model, data)

    last = time.time()
    step = 0
    
    # Recovery tracking
    recovery_start_time = None
    
    while viewer.is_running():
        now = time.time()
        dt = now - last
        last = now

        # ========== READ SENSOR DATA ==========
        pend_angle = data.qpos[pend_qpos_idx]
        pend_vel = data.qvel[pend_qvel_idx]
        
        # Wrap angle to [-pi, pi] for control
        pend_angle_wrapped = (pend_angle + np.pi) % (2*np.pi) - np.pi
        
        # Wheel velocities for damping
        left_wheel_vel = data.qvel[left_wheel_dof]
        right_wheel_vel = data.qvel[right_wheel_dof]
        avg_wheel_vel = (left_wheel_vel + right_wheel_vel) / 2.0

        # ========== STABILITY DETECTION ==========
        is_stable = (abs(pend_angle_wrapped) < STABILITY_ANGLE_THRESHOLD and 
                     abs(pend_vel) < STABILITY_VEL_THRESHOLD)
        
        if is_stable:
            if stable_start_time is None:
                stable_start_time = now
                print(f"[Stability] Starting stability timer at {now:.2f}s")
        else:
            stable_start_time = None

        # ========== DISTURBANCE LOGIC ==========
        apply_disturbance = False
        if (stable_start_time is not None and 
            now - stable_start_time >= STABLE_TIME_REQUIRED and
            now >= next_disturbance_time):
            
            # Calculate disturbance strength based on how stable we are
            stability_factor = min(1.0, (now - stable_start_time) / (STABLE_TIME_REQUIRED * 2))
            disturbance_strength = DISTURBANCE_STRENGTH_MIN + \
                                  (DISTURBANCE_STRENGTH_MAX - DISTURBANCE_STRENGTH_MIN) * stability_factor
            
            # Apply random disturbance
            disturbance_sign = np.random.choice([-1, 1])
            disturbance = disturbance_sign * disturbance_strength
            
            # Apply as angular velocity kick
            data.qvel[pend_qvel_idx] += disturbance
            
            # Reset PID integral to prevent windup after disturbance
            pid_controller.reset_integral()
            
            # Update tracking
            disturbance_count += 1
            apply_disturbance = True
            
            print(f"[DISTURBANCE #{disturbance_count}] Applied {disturbance:.3f} rad/s kick")
            print(f"  Stability factor: {stability_factor:.2f}")
            print(f"  Pre-disturbance: angle={pend_angle_wrapped:.4f} rad, vel={pend_vel:.4f} rad/s")
            
            # Reset stability timer
            stable_start_time = None
            
            # Schedule next disturbance
            next_disturbance_time = now + np.random.uniform(DISTURBANCE_INTERVAL_MIN, DISTURBANCE_INTERVAL_MAX)
            
            # Start recovery timer
            recovery_start_time = now

        # ========== CONTROL LOGIC ==========
        control_active = False
        wheel_torque = 0.0
        
        # Check if we need to apply control
        needs_control = (abs(pend_angle_wrapped) > CONTROL_ANGLE_THRESHOLD or 
                         abs(pend_vel) > CONTROL_VEL_THRESHOLD)
        
        if needs_control:
            # Calculate control error (negative for corrective action)
            angle_error = -pend_angle_wrapped
            
            # Get PID output
            base_torque = pid_controller.update(angle_error, pend_angle_wrapped, pend_vel)
            
            # Apply control sign
            wheel_torque = CONTROL_SIGN * base_torque
            
            # Add wheel velocity damping (braking)
            wheel_torque -= WHEEL_DAMPING * avg_wheel_vel
            
            # Clip torque to limits
            wheel_torque = float(np.clip(wheel_torque, -MAX_WHEEL_TORQUE, MAX_WHEEL_TORQUE))
            
            control_active = True
            
            # Track recovery if we're recovering from disturbance
            if recovery_start_time is not None:
                recovery_time = now - recovery_start_time
                if is_stable:
                    recovery_times.append(recovery_time)
                    print(f"[RECOVERY] Stabilized in {recovery_time:.2f}s after disturbance")
                    recovery_start_time = None
        else:
            # Minimal damping when near equilibrium
            wheel_torque = -0.1 * avg_wheel_vel
            control_active = False

        # ========== APPLY CONTROL ==========
        if use_actuators:
            # Use actuator control (for models with motor actuators)
            data.ctrl[left_motor_idx] = wheel_torque
            data.ctrl[right_motor_idx] = wheel_torque
        else:
            # Use direct joint control
            data.ctrl[left_motor_idx] = wheel_torque
            data.ctrl[right_motor_idx] = wheel_torque

        # ========== STEP SIMULATION ==========
        mujoco.mj_step(model, data)
        viewer.sync()

        # ========== STATUS UPDATES ==========
        if step % 50 == 0:  # Update every ~0.1s at 500Hz
            status = "STABLE" if is_stable else "UNSTABLE"
            control_status = "ACTIVE" if control_active else "MINIMAL"
            
            # Calculate time until next disturbance
            time_until_disturbance = max(0, next_disturbance_time - now)
            
            # Calculate stable time if applicable
            if stable_start_time is not None:
                stable_time = now - stable_start_time
                stability_progress = stable_time / STABLE_TIME_REQUIRED
            else:
                stable_time = 0
                stability_progress = 0
            
            # Calculate recovery time if recovering
            recovery_info = ""
            if recovery_start_time is not None:
                recovery_elapsed = now - recovery_start_time
                recovery_info = f" | Recovery: {recovery_elapsed:.2f}s"
            
            print(f"t={now:.2f}s | Angle={pend_angle_wrapped:.4f} | Vel={pend_vel:.4f} | "
                  f"Status={status} | Control={control_status}{recovery_info}")
            
            if stable_time > 0:
                print(f"  Stable for: {stable_time:.1f}/{STABLE_TIME_REQUIRED}s "
                      f"({stability_progress*100:.0f}%)")
            
            if time_until_disturbance > 0 and stable_start_time is not None:
                print(f"  Next disturbance in: {time_until_disturbance:.1f}s")
        
        step += 1

# ========== FINAL STATISTICS ==========
print("\n" + "="*50)
print("SIMULATION SUMMARY")
print("="*50)
print(f"Total disturbances applied: {disturbance_count}")
if recovery_times:
    print(f"Average recovery time: {np.mean(recovery_times):.2f}s")
    print(f"Best recovery time: {min(recovery_times):.2f}s")
    print(f"Worst recovery time: {max(recovery_times):.2f}s")
else:
    print("No successful recoveries recorded")
print("="*50)