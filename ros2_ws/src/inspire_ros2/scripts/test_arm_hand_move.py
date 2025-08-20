import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
import numpy as np
import rtde_receive, rtde_control
from sensor_msgs.msg import JointState
from inspire_ros2.common import JOINT_NAMES, JOINT_MIN, JOINT_MAX, INDEX_EXTEND_ONLY_POS


class RTDEClient(object):
    def __init__(self, robot_ip='192.168.54.130'):
        self.robot_ip = robot_ip
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.rtde_c = rtde_control.RTDEControlInterface("192.168.54.130", frequency=500, flags=rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP, ur_cap_port=50002)

    def get_joints(self):
        return self.rtde_r.getActualQ()
    
    def move_joints(self, q):
        self.rtde_c.moveJ(q, speed=0.2, acceleration=0.5, asynchronous=True)

    def move_joints_continuous(self, q, dt):
        t_start = self.rtde_c.initPeriod()
        self.rtde_c.servoJ(q, 0.5, 0.5, dt, 0.03, 300)
        self.rtde_c.waitPeriod(t_start)


class ArmHandCmdPublisher(Node):
    def __init__(self):
        super().__init__('arm_hand_cmd_publisher')
        self.arm = RTDEClient()
        breakpoint()

        self.publisher_ = self.create_publisher(JointState, '/inspire/joint_cmd', 10)
        self.ur_js_pub = self.create_publisher(JointState, '/ur/joint_states', 10)

        self.ctrl_cbg = MutuallyExclusiveCallbackGroup()
        self.pub_cbg = MutuallyExclusiveCallbackGroup()

        self.joint_names = JOINT_NAMES
        self.joint_min = JOINT_MIN
        self.joint_max = JOINT_MAX

        self.ur_joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        self.arm_q0 = np.array([3.6866016387939453, -0.24869043031801397, 1.494713306427002, -2.5008228460894983, 5.19681978225708, -2.533320967351095])
        self.arm_q_range = np.array([0.1, 0.05, 0.1, 0.2, 0.2, 0.2])
        self.arm.move_joints(self.arm_q0)

        # initialize
        self.hand_goto(INDEX_EXTEND_ONLY_POS, 1.0*np.ones(len(INDEX_EXTEND_ONLY_POS)))
        self.dt = 0.025
        # self.dt = 0.1
        self.T = 10

        time.sleep(2.0)

        self.t0 = time.time()
        self.ctrl_timer = self.create_timer(self.dt, self.control_loop, callback_group=self.ctrl_cbg)
        self.pub_timer = self.create_timer(1 / 50, self.publish_loop, callback_group=self.pub_cbg)

    def hand_goto(self, q, v):
        joint_state = JointState()
        joint_state.name = self.joint_names
        joint_state.position = q.tolist()
        joint_state.velocity = v.tolist()
        self.publisher_.publish(joint_state)

    def publish_loop(self):
        ur_joint_state = JointState()
        ur_joint_state.name = self.ur_joint_names
        ur_joint_state.position = self.arm.get_joints()
        ur_joint_state.header.stamp = self.get_clock().now().to_msg()
        self.ur_js_pub.publish(ur_joint_state)

    def control_loop(self):
        t = time.time()
        index_min, index_max = JOINT_MIN[2], JOINT_MAX[2] - 0.2

        # index_pos = 0.5 * (index_min + index_max) + 0.5 * (index_max - index_min) * np.sin(2 * np.pi * (t - self.t0) / self.T - np.pi / 2)
        # index_vel = abs(np.cos(2 * np.pi * (t - self.t0) / self.T - np.pi / 2))
        
        if (t - self.t0) % (2 * self.T) < self.T:
            index_pos = index_max
        else:
            index_pos = index_min
        index_vel = 0.06
        
        curr_finger_pos = INDEX_EXTEND_ONLY_POS.copy()
        curr_finger_pos[2] = index_pos
        curr_finger_vel = np.zeros(len(INDEX_EXTEND_ONLY_POS))
        curr_finger_vel[2] = index_vel
        self.hand_goto(curr_finger_pos, curr_finger_vel)

        curr_arm_pos = self.arm_q0 + 0.5 * self.arm_q_range * np.sin(2 * np.pi * (t - self.t0) / self.T)
        self.arm.move_joints_continuous(curr_arm_pos, 0.1)

        self.get_logger().info(f'Index position: {index_pos}, velocity: {index_vel}')

        self.get_logger().info(f'time: {t - self.t0:.2f}')
        

def main(args=None):
    rclpy.init(args=args)
    node = ArmHandCmdPublisher()
    rclpy.spin(node)
    node.arm.move_joints(node.arm_q0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
