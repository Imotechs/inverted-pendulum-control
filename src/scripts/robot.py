
import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "xml/slider.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
with mujoco.viewer.launch_passive(model, data) as viewer:

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    last_time = time.time()
    counter = 0
    while viewer.is_running():

        now = time.time()
        dt = now - last_time
        last_time = now

        counter += 1
        mujoco.mj_step(model, data)

        # update viewer
        viewer.sync()