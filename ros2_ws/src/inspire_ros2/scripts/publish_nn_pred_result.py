import os
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle


def matrix_to_quat(matrix):
    return R.from_matrix(matrix).as_quat().tolist()

def float_to_ros_time(t: float) -> Time:
    msg = Time()
    msg.sec = int(t)  # integer seconds
    msg.nanosec = int((t - msg.sec) * 1e9)  # fractional part → nanoseconds
    return msg

class PoseLogger(Node):
    def __init__(self):
        super().__init__('pose_logger')

        self.pose_key = 'ftip_pose'
        bag_path = "/home/jyp/hardware/inspire_hand/exp_data/20250820/rosbag2_2025_08_20-14_32_21"
        inference_path = os.path.join(bag_path, "parsed", f"full_dataset.pkl")
        self.dataset = pickle.load(open(inference_path, "rb"))

        # Publishers (these will be recorded by rosbag)
        self.pub_gt = self.create_publisher(PoseStamped, '/ee_pose/ground_truth', 10)
        self.pub_fk = self.create_publisher(PoseStamped, '/ee_pose/forward_kinematics', 10)
        self.pub_pred = self.create_publisher(PoseStamped, '/ee_pose/predicted', 10)

        self.cur_idx = 0
        self.timer = self.create_timer(1 / 30, self.timer_callback)  # 30Hz

    def publish_pose(self, stamp, pose_gt, pose_fk, pose_pred):
        """
        pose_gt, pose_fk, pose_pred: dict with keys { 'pos': np.array(3,), 'quat': np.array(4,) }
        """
        msg_gt = PoseStamped()
        msg_gt.header.stamp = stamp
        msg_gt.header.frame_id = "world"
        msg_gt.pose.position.x, msg_gt.pose.position.y, msg_gt.pose.position.z = pose_gt['pos']
        msg_gt.pose.orientation.x, msg_gt.pose.orientation.y, msg_gt.pose.orientation.z, msg_gt.pose.orientation.w = pose_gt['quat']

        msg_fk = PoseStamped()
        msg_fk.header.stamp = stamp
        msg_fk.header.frame_id = "world"
        msg_fk.pose.position.x, msg_fk.pose.position.y, msg_fk.pose.position.z = pose_fk['pos']
        msg_fk.pose.orientation.x, msg_fk.pose.orientation.y, msg_fk.pose.orientation.z, msg_fk.pose.orientation.w = pose_fk['quat']

        msg_pred = PoseStamped()
        msg_pred.header.stamp = stamp
        msg_pred.header.frame_id = "world"
        msg_pred.pose.position.x, msg_pred.pose.position.y, msg_pred.pose.position.z = pose_pred['pos']
        msg_pred.pose.orientation.x, msg_pred.pose.orientation.y, msg_pred.pose.orientation.z, msg_pred.pose.orientation.w = pose_pred['quat']

        self.pub_gt.publish(msg_gt)
        self.pub_fk.publish(msg_fk)
        self.pub_pred.publish(msg_pred)

    def timer_callback(self):
        if self.cur_idx >= len(self.dataset['t']):
            self.timer.cancel()
            return
        
        if self.cur_idx % 30 == 0:
            self.get_logger().info(f"Publishing pose at index {self.cur_idx}/{len(self.dataset['t'])}")

        t = float_to_ros_time(self.dataset['t'][self.cur_idx])
        pose_gt = {
            'pos': np.array(self.dataset[self.pose_key]['gt'][self.cur_idx])[:3, 3],
            'quat': matrix_to_quat(np.array(self.dataset['ftip_pose']['gt'][self.cur_idx])[:3, :3])
        }
        pose_fk = {
            'pos': np.array(self.dataset[self.pose_key]['pred'][self.cur_idx])[:3, 3],
            'quat': matrix_to_quat(np.array(self.dataset['ftip_pose']['pred'][self.cur_idx])[:3, :3])
        }
        pose_pred = {
            'pos': np.array(self.dataset[self.pose_key]['nn_pred'][self.cur_idx])[:3, 3],
            'quat': matrix_to_quat(np.array(self.dataset['ftip_pose']['nn_pred'][self.cur_idx])[:3, :3])
        }

        self.publish_pose(t, pose_gt, pose_fk, pose_pred)
        self.cur_idx += 1


if __name__ == '__main__':
    rclpy.init()
    node = PoseLogger()
    rclpy.spin(node)
    rclpy.shutdown()
