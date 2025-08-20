#!/usr/bin/python

import os
import sys
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
sys.path.append('/home/jyp/research/key_research/inspire_hand_ws/inspire_hand_sdk/example')
from inspire_sdkpy import inspire_hand_defaut,inspire_dds
from dds_subscribe import DDSHandler
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from inspire_ros2.utils import get_joint_state_msg, map_joint_to_actuator, map_joint_velocity_to_actuator


class InspireHandNode(Node):
    def __init__(self):
        super().__init__('inspire_hand_node')
        self.get_logger().info('Node has been started')

        self.dds_handler = DDSHandler(LR='r')
        self.pubr = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_dds.inspire_hand_ctrl)
        self.pubr.Init()
        self.cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        # self.cmd.mode=0b0001  # ANGLE_SET
        self.cmd.mode=0b1001    # ANGLE_SET + SPEED_SET

        self.joint_state_pub = self.create_publisher(
            JointState,
            '/inspire/joint_states',
            10
        )

        self.joint_cmd_sub = self.create_subscription(
            JointState,
            '/inspire/joint_cmd',
            self.joint_cmd_callback,
            10
        )

        self.get_logger().info('Subscriptions and publishers have been set up')
        self.timer = self.create_timer(1/50, self.timer_callback)

    def timer_callback(self):
        data = self.dds_handler.read()
        # self.get_logger().info(f"Joint states: {data['states']}")

        # read from register: pinky, ring, middle, index, thumb, thumb_base
        joint_pos = data['states']['ANGLE_ACT']
        # urdf order: thumb_base, thumb, index, middle, ring, pinky
        joint_pos_reverse = joint_pos[::-1]
        joint_state_msg = get_joint_state_msg(joint_pos_reverse)
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        self.joint_state_pub.publish(joint_state_msg)

    def joint_cmd_callback(self, msg:JointState):
        actuator_values = map_joint_to_actuator(msg.name, msg.position)
        actuator_vels = map_joint_velocity_to_actuator(msg.name, msg.velocity)
        self.cmd.angle_set = actuator_values
        self.cmd.speed_set = actuator_vels
        self.pubr.Write(self.cmd)
        self.get_logger().info(f"ANGLE_SET sent: {self.cmd.angle_set}")
        self.get_logger().info(f"SPEED_SET sent: {self.cmd.speed_set}")
        

def main(args=None):
    rclpy.init(args=args)
    node = InspireHandNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()