#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import JointState
import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
from pinocchio.visualize import MeshcatVisualizer

from inspire_ros2.common import JOINT_NAMES, PASSIVE_JOINTS, FINGER_TIP_NAMES
from inspire_ros2.utils import get_q_from_act_joints


class InspirePinocchio(Node):
    def __init__(self):
        super().__init__('inspire_pinocchio')
        self.get_logger().info('Node has been started')

        hand_model_root = "/home/jyp/research/key_research/inspire_hand_ws/ros2_ws/src/inspire_hand_description"
        hand_urdf_url = os.path.join(hand_model_root, "urdf", "inspire_hand_right_pinocchio.urdf")
        hand_mesh_url = os.path.join(hand_model_root, "meshes", "right")

        # 加载模型
        pin_model, collision_model, visual_model = pin.buildModelsFromUrdf(hand_urdf_url, hand_mesh_url)
        data = pin_model.createData()

        self.pin_model, self.data = pin_model, data

        # 创建 Meshcat 可视化器
        visualizer = MeshcatVisualizer(pin_model, collision_model, visual_model)
        visualizer.initViewer()
        visualizer.loadViewerModel()

        self.visualizer = visualizer

        self.fk_from_model_pub = self.create_publisher(
            PoseArray,
            '/fk_from_model',
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

    def joint_state_callback(self, msg:JointState):
        act_joints = {j:q for j, q in zip(msg.name, msg.position)}
        all_joints = get_q_from_act_joints(self.pin_model, act_joints)
        self.visualizer.display(all_joints)

        # compute forward kinematics of fingertips
        pin.forwardKinematics(self.pin_model, self.data, all_joints)
        pin.updateFramePlacements(self.pin_model, self.data)

        # get the pose of the end effector
        finger_tip_fk = PoseArray()
        for tip in FINGER_TIP_NAMES:
            tip_id = self.pin_model.getFrameId(tip)
            tip_pose = self.data.oMf[tip_id]
            tip_position = tip_pose.translation
            tip_rotation = tip_pose.rotation
            tip_quaternion = pin.Quaternion(tip_rotation)

            pose_msg = Pose()
            pose_msg.position.x = tip_position[0]
            pose_msg.position.y = tip_position[1]
            pose_msg.position.z = tip_position[2]
            pose_msg.orientation.x = tip_quaternion.x
            pose_msg.orientation.y = tip_quaternion.y
            pose_msg.orientation.z = tip_quaternion.z
            pose_msg.orientation.w = tip_quaternion.w

            finger_tip_fk.poses.append(pose_msg)
        finger_tip_fk.header.stamp = self.get_clock().now().to_msg()
        finger_tip_fk.header.frame_id = "LINK_ROOT"
        self.fk_from_model_pub.publish(finger_tip_fk)
        

def main(args=None):
    rclpy.init(args=args)
    node = InspirePinocchio()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
