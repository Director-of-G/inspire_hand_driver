import rtde_control
import rtde_receive
import numpy as np
import time

rtde_c = rtde_control.RTDEControlInterface("192.168.54.130", frequency=500, flags=rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP, ur_cap_port=50002)
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.54.130")

# Parameters
velocity = 0.5
acceleration = 0.5
dt = 1.0/500  # 2ms
lookahead_time = 0.1
gain = 300

q0 = np.array(rtde_r.getActualQ())  # Get current joint positions
breakpoint()
arm_q_range = np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05])

# Move to initial joint position with a regular moveJ
rtde_c.moveJ(q0)

# Execute 500Hz control loop for 2 seconds, each cycle is 2ms
t0 = time.time()
T = 20
for i in range(10000):
    t = time.time()
    t_start = rtde_c.initPeriod()
    new_joints = q0 + 0.5 * arm_q_range * np.sin(2 * np.pi * (t - t0) / T)
    rtde_c.servoJ(new_joints, velocity, acceleration, dt, lookahead_time, gain)
    tttt = time.time()
    rtde_c.waitPeriod(t_start)
    print(f"Servoing at time: {time.time() - tttt:.3f}s")

rtde_c.servoStop()
rtde_c.stopScript()