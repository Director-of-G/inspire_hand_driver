# Copyright 2020 Yutaka Kondo <yutaka.kondo@youtalk.jp>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

import xacro


def generate_launch_description():
    robot_name = "inspire_hand_right"
    package_name = "inspire_hand_description"
    rviz_config = os.path.join(get_package_share_directory(
        package_name), "rviz", robot_name + ".rviz")
    robot_description = os.path.join(get_package_share_directory(
        package_name), "urdf", robot_name + ".urdf")
    robot_description_config = xacro.process_file(robot_description)

    # 声明 use_gui 参数
    use_gui_arg = DeclareLaunchArgument(
        'use_gui',
        default_value='true',
        description='Whether to start joint_state_publisher_gui'
    )

    # 创建节点列表
    launch_list = [use_gui_arg]

    # use GUI
    launch_list.append(
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            parameters=[{'robot_description': robot_description_config.toxml()}],
            condition=IfCondition(LaunchConfiguration('use_gui'))
        )
    )

    # use hardware when use_gui is false
    launch_list.append(
        Node(
            package="inspire_ros2",
            executable="inspire_hand_node.py",
            name="robot_state_publisher",
            output="screen",
            condition=UnlessCondition(LaunchConfiguration('use_gui'))
        )
    )

    launch_list.extend([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[
                {"robot_description": robot_description_config.toxml()}],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            parameters=[
                {"robot_description": robot_description_config.toxml()}],
            arguments=["-d", rviz_config],
            output="screen",
        ),
        Node(
            package="inspire_ros2",
            executable="inspire_pinocchio.py",
            name="robot_state_publisher",
            output="screen",
        ),
    ])

    return LaunchDescription(launch_list)