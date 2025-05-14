import rclpy
from rclpy.node import Node
import time
from sensor_msgs.msg import JointState
from inspire_ros2.common import JOINT_NAMES, JOINT_MIN, JOINT_MAX

class JointCmdPublisher(Node):
    def __init__(self):
        super().__init__('joint_cmd_publisher')
        self.publisher_ = self.create_publisher(JointState, '/joint_cmd', 10)
        self.joint_names = JOINT_NAMES
        self.joint_min = JOINT_MIN
        self.joint_max = JOINT_MAX

        self.timer = self.create_timer(0.3, self.control_joints)
        self.current_joint_positions = list(self.joint_min)
        self.direction = 1
        self.current_joint_index = 0

        # initialize
        joint_state = JointState()
        joint_state.name = self.joint_names
        joint_state.position = self.current_joint_positions
        self.publisher_.publish(joint_state)
        self.get_logger().info(f'Published initial JointState: {joint_state.position}')

        time.sleep(2.0)

    def control_joints(self):
        joint_state = JointState()
        joint_state.name = self.joint_names
        joint_state.position = self.current_joint_positions

        self.publisher_.publish(joint_state)

        # Update the current joint position
        self.current_joint_positions[self.current_joint_index] += self.direction * 0.1

        # Check if the current joint has reached its limit
        if self.direction == 1 and self.current_joint_positions[self.current_joint_index] >= self.joint_max[self.current_joint_index]:
            self.direction = -1
        elif self.direction == -1 and self.current_joint_positions[self.current_joint_index] <= self.joint_min[self.current_joint_index]:
            self.direction = 1
            self.current_joint_index = (self.current_joint_index + 1) % len(self.joint_names)

def main(args=None):
    rclpy.init(args=args)
    node = JointCmdPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()