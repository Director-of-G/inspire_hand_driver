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


hand_model_root = "/home/jyp/hardware/inspire_hand/inspire_hand_driver/ros2_ws/src/inspire_hand_description"
hand_urdf_url = os.path.join(hand_model_root, "urdf", "ur5_inspire_right_pinocchio.urdf")
hand_mesh_url = os.path.join(hand_model_root, "meshes")

# 加载模型
pin_model, collision_model, visual_model = pin.buildModelsFromUrdf(hand_urdf_url, hand_mesh_url)
data = pin_model.createData()

# joint order
# shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint
# right_index_1_joint, right_index_2_joint,
# right_little_1_joint, right_little_2_joint,
# right_middle_1_joint, right_middle_2_joint,
# right_ring_1_joint, right_ring_2_joint,
# right_thumb_1_joint, right_thumb_2_joint, right_thumb_3_joint, right_thumb_4_joint
pin_model, data = pin_model, data

# 创建 Meshcat 可视化器
visualizer = MeshcatVisualizer(pin_model, collision_model, visual_model)
visualizer.initViewer()
visualizer.loadViewerModel()

q0 = np.zeros(6+12)
visualizer.display(q0)

breakpoint()
