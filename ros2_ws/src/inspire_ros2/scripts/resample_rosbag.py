import os
import numpy as np
import pickle


bag_path = "/home/jyp/hardware/inspire_hand/exp_data/20250820/rosbag2_2025_08_20-14_32_21"
poses = pickle.load(open(os.path.join(bag_path, "parsed", "poses.pkl"), "rb"))
ur_js = pickle.load(open(os.path.join(bag_path, "parsed", "ur_js.pkl"), "rb"))
inspire_js = pickle.load(open(os.path.join(bag_path, "parsed", "inspire_js.pkl"), "rb"))

desire_inspire_jname_order = [
    'right_index_1_joint', 'right_index_2_joint',
    'right_little_1_joint', 'right_little_2_joint',
    'right_middle_1_joint', 'right_middle_2_joint',
    'right_ring_1_joint', 'right_ring_2_joint',
    'right_thumb_1_joint', 'right_thumb_2_joint', 'right_thumb_3_joint', 'right_thumb_4_joint'
]

inspire_jname_remap = {
    'J_THUMB_PROXIMAL_BASE_R': 'right_thumb_1_joint',
    'J_THUMB_PROXIMAL_R': 'right_thumb_2_joint',
    'J_THUMB_INTERMEDIATE_R': 'right_thumb_3_joint',
    'J_THUMB_DISTAL_R': 'right_thumb_4_joint',
    'J_INDEX_PROXIMAL_R': 'right_index_1_joint',
    'J_INDEX_INTERMEDIATE_R': 'right_index_2_joint',
    'J_MIDDLE_PROXIMAL_R': 'right_middle_1_joint',
    'J_MIDDLE_INTERMEDIATE_R': 'right_middle_2_joint',
    'J_RING_PROXIMAL_R': 'right_ring_1_joint',
    'J_RING_INTERMEDIATE_R': 'right_ring_2_joint',
    'J_PINKY_PROXIMAL_R': 'right_little_1_joint',
    'J_PINKY_INTERMEDIATE_R': 'right_little_2_joint'
}

aligned_ur_js = []
aligned_inspire_js = []

t_ur = np.array(ur_js['t'])
t_inspire = np.array(inspire_js['t'])
record_inspire_jname = [inspire_jname_remap[name] for name in inspire_js['names']]
inspire_js_remap = [desire_inspire_jname_order.index(name) for name in record_inspire_jname]
breakpoint()

for t_sec in poses['t']:
    idx_ur = np.argmin(np.abs(t_ur - t_sec))
    idx_inspire = np.argmin(np.abs(t_inspire - t_sec))
    
    aligned_ur_js.append(ur_js['q'][idx_ur])
    aligned_inspire_js.append(inspire_js['q'][idx_inspire])

dataset = {
    "t": poses['t'],
    "poses": poses['poses'],
    "ur_js": aligned_ur_js,
    "inspire_js": np.array(aligned_inspire_js)[:, inspire_js_remap].tolist()
}

pickle.dump(dataset, open(os.path.join(bag_path, "parsed", "dataset.pkl"), "wb"))

breakpoint()
