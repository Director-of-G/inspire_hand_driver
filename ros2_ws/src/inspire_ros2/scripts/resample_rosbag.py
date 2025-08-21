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
inspire_js_remap = [record_inspire_jname.index(name) for name in desire_inspire_jname_order]

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

# train test split
full_length = len(dataset['t'])
train_ratio = 0.8
train_idx = np.random.choice(np.arange(full_length), size=int(train_ratio * full_length), replace=False)
train_idx = np.sort(train_idx)
test_idx = np.setdiff1d(np.arange(full_length), train_idx)

trainset = {
    "t": np.array(dataset['t'])[train_idx].tolist(),
    "poses": {id: np.array(dataset['poses'][id])[train_idx].tolist() for id in dataset['poses']},
    "ur_js": np.array(dataset['ur_js'])[train_idx].tolist(),
    "inspire_js": np.array(dataset['inspire_js'])[train_idx].tolist()
}
testset = {
    "t": np.array(dataset['t'])[test_idx].tolist(),
    "poses": {id: np.array(dataset['poses'][id])[test_idx].tolist() for id in dataset['poses']},
    "ur_js": np.array(dataset['ur_js'])[test_idx].tolist(),
    "inspire_js": np.array(dataset['inspire_js'])[test_idx].tolist()
}

pickle.dump(dataset, open(os.path.join(bag_path, "parsed", "full_dataset.pkl"), "wb"))
pickle.dump(trainset, open(os.path.join(bag_path, "parsed", "train_dataset.pkl"), "wb"))
pickle.dump(testset, open(os.path.join(bag_path, "parsed", "test_dataset.pkl"), "wb"))

