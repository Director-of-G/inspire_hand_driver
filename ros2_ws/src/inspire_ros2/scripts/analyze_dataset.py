import os
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
import pickle


T_ur2marker0 = np.array([
    [-1, 0, 0, -0.06],
    [0, 0, -1, -0.355],
    [0, -1, 0, -0.198],
    [0, 0, 0, 1]
])

T_marker2tip = np.array([
    [0, 0, 1, 0.025],
    [0, 1, 0, 0.065],
    [-1, 0, 0, -0.005],
    [0, 0, 0, 1]
])

T_marker2wrist = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

def pose_arr_to_matrix(arr):
    arr = np.array(arr)
    matrix = np.eye(4)
    matrix[:3, 3] = arr[:3]
    matrix[:3, :3] = R.from_quat(arr[3:][[1, 2, 3, 0]]).as_matrix()
    return matrix

def rotation_matrix_to_axis_angle(mat):
    return R.from_matrix(mat).as_rotvec()

dataset_name = 'full_dataset'
bag_path = "/home/jyp/hardware/inspire_hand/exp_data/20250820/rosbag2_2025_08_20-14_32_21"
dataset = pickle.load(open(os.path.join(bag_path, "parsed", f"{dataset_name}.pkl"), "rb"))
dataset['ftip_pose'] = {
    "pred": [],
    "gt": []
}
dataset['wrist_pose'] = {
    "pred": [],
    "gt": []
}

model_root = "/home/jyp/hardware/inspire_hand/inspire_hand_driver/ros2_ws/src/inspire_hand_description"
urdf_url = os.path.join(model_root, "urdf", "ur5_inspire_right_pinocchio.urdf")
mesh_url = os.path.join(model_root, "meshes")

pin_model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_url, mesh_url)
data = pin_model.createData()

frames = [frame.name for frame in pin_model.frames]
index_tip_idx = frames.index('right_index_2')
wrist_idx = frames.index('inspire_base_link')

trans_pred_error = []
ori_pred_error = []

for i in range(len(dataset['t'])):
    ur_marker_pose = pose_arr_to_matrix(dataset['poses'][0][i])
    ur_base_pose = ur_marker_pose @ T_ur2marker0

    full_q = np.array(dataset['ur_js'][i] + dataset['inspire_js'][i])
    pin.forwardKinematics(pin_model, data, full_q)

    pin.updateFramePlacements(pin_model, data)
    ftip_pose = data.oMf[index_tip_idx].np.copy()
    ftip_pose = ur_base_pose @ ftip_pose @ T_marker2tip

    ftip_pose_gt = pose_arr_to_matrix(dataset['poses'][10][i])
    trans_pred_error.append(np.linalg.norm(ftip_pose[:3, 3] - ftip_pose_gt[:3, 3]))

    ori_pred_error.append(np.linalg.norm(rotation_matrix_to_axis_angle(ftip_pose[:3, :3].T @ ftip_pose_gt[:3, :3])))

    dataset['ftip_pose']['pred'].append(ftip_pose)
    dataset['ftip_pose']['gt'].append(ftip_pose_gt)

print("------ ftip pose ------")
print(f"trans error: min={np.min(trans_pred_error)}, max={np.max(trans_pred_error)}, avg={np.mean(trans_pred_error)}")
print(f"ori error: min={np.min(ori_pred_error)}, max={np.max(ori_pred_error)}, avg={np.mean(ori_pred_error)}")

dataset['ftip_pose']['pred'] = np.array(dataset['ftip_pose']['pred'])
dataset['ftip_pose']['gt'] = np.array(dataset['ftip_pose']['gt'])

for i in range(len(dataset['t'])):
    ur_marker_pose = pose_arr_to_matrix(dataset['poses'][0][i])
    ur_base_pose = ur_marker_pose @ T_ur2marker0

    full_q = np.array(dataset['ur_js'][i] + dataset['inspire_js'][i])
    pin.forwardKinematics(pin_model, data, full_q)

    pin.updateFramePlacements(pin_model, data)
    wrist_pose = data.oMf[wrist_idx].np.copy()
    wrist_pose = ur_base_pose @ wrist_pose @ T_marker2wrist

    wrist_pose_gt = pose_arr_to_matrix(dataset['poses'][20][i])
    trans_pred_error.append(np.linalg.norm(wrist_pose[:3, 3] - wrist_pose_gt[:3, 3]))

    ori_pred_error.append(np.linalg.norm(rotation_matrix_to_axis_angle(wrist_pose[:3, :3].T @ wrist_pose_gt[:3, :3])))

    dataset['wrist_pose']['pred'].append(wrist_pose)
    dataset['wrist_pose']['gt'].append(wrist_pose_gt)

print("------ wrist pose ------")
print(f"trans error: min={np.min(trans_pred_error)}, max={np.max(trans_pred_error)}, avg={np.mean(trans_pred_error)}")
print(f"ori error: min={np.min(ori_pred_error)}, max={np.max(ori_pred_error)}, avg={np.mean(ori_pred_error)}")

dataset['wrist_pose']['pred'] = np.array(dataset['wrist_pose']['pred'])
dataset['wrist_pose']['gt'] = np.array(dataset['wrist_pose']['gt'])

pickle.dump(dataset, open(os.path.join(bag_path, "parsed", f"{dataset_name}.pkl"), "wb"))
