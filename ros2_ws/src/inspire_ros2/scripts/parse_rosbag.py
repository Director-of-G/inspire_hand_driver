import rosbag2_py
import rclpy
from rclpy.serialization import deserialize_message
import os
import numpy as np
import pickle
from geometry_msgs.msg import Pose
from rosidl_runtime_py.utilities import get_message

def read_rosbag(path, topics=None):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {}
    for conn in reader.get_all_topics_and_types():
        type_map[conn.name] = get_message(conn.type)

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topics and topic not in topics:
            continue
        msg_type = type_map[topic]
        msg = deserialize_message(data, msg_type)
        yield topic, msg, t   # t = nanoseconds

def pos_msg_to_array(msg: Pose):
    arr = [msg.position.x, msg.position.y, msg.position.z,
           msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
    return arr

# Example: sample messages at fixed 20Hz

bag_path = "/home/jyp/hardware/inspire_hand/exp_data/20250820/rosbag2_2025_08_20-14_32_21"
flag_parse_poses = True
flag_parse_ur_js = False
flag_parse_inspire_js = False
os.makedirs(os.path.join(bag_path, "parsed"), exist_ok=True)

if flag_parse_poses:
    frame_ids = [0, 10, 20]
    parsed_poses = {
        "t": [],
        "poses": {id: [] for id in frame_ids}
    }
    for topic, msg, t in read_rosbag(bag_path, topics=["/apriltag_detections"]):
        t_sec = t * 1e-9
        parsed_poses["t"].append(t_sec)
        detect_ids = [det.id for det in msg.detections]
        for id in frame_ids:
            if id in detect_ids:
                idx = detect_ids.index(id)
                pose = msg.detections[idx].pose.pose.pose
                parsed_poses["poses"][id].append(pos_msg_to_array(pose))
            else:
                parsed_poses["poses"][id].append([np.nan] * 7)
    with open(os.path.join(bag_path, "parsed", "poses.pkl"), "wb") as f:
        pickle.dump(parsed_poses, f)

if flag_parse_ur_js:
    parsed_ur_js = {
        "t": [],
        "names": [],
        "q": []
    }
    for topic, msg, t in read_rosbag(bag_path, topics=["/ur/joint_states"]):
        t_sec = t * 1e-9
        parsed_ur_js["t"].append(t_sec)
        if parsed_ur_js["names"] == []:
            parsed_ur_js["names"] = msg.name
        joint_state = np.array(msg.position).tolist()
        parsed_ur_js["q"].append(joint_state)
    with open(os.path.join(bag_path, "parsed", "ur_js.pkl"), "wb") as f:
        pickle.dump(parsed_ur_js, f)

if flag_parse_inspire_js:
    parsed_inspire_js = {
        "t": [],
        "names": [],
        "q": []
    }
    for topic, msg, t in read_rosbag(bag_path, topics=["/inspire/joint_states"]):
        t_sec = t * 1e-9
        parsed_inspire_js["t"].append(t_sec)
        if parsed_inspire_js["names"] == []:
            parsed_inspire_js["names"] = msg.name
        joint_state = np.array(msg.position).tolist()
        parsed_inspire_js["q"].append(joint_state)
    with open(os.path.join(bag_path, "parsed", "inspire_js.pkl"), "wb") as f:
        pickle.dump(parsed_inspire_js, f)
