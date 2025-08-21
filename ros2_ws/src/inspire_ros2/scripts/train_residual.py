import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Utility: axis-angle <-> rotmat
# -----------------------------
def axis_angle_to_matrix(axis_angle):
    """Convert axis-angle vector to rotation matrix."""
    theta = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis_angle / theta
    K = torch.zeros(axis.shape[0], 3, 3, device=axis.device)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    K[:, 0, 1], K[:, 0, 2] = -z, y
    K[:, 1, 0], K[:, 1, 2] = z, -x
    K[:, 2, 0], K[:, 2, 1] = -y, x
    I = torch.eye(3, device=axis.device).unsqueeze(0)
    R = I + torch.sin(theta)[..., None] * K + (1 - torch.cos(theta))[..., None] * (K @ K)
    return R


def rotation_geodesic_loss(R_pred, R_gt):
    """Geodesic distance between two rotation matrices."""
    R_rel = R_pred.transpose(-1, -2) @ R_gt
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos_theta)


# -----------------------------
# Dataset
# -----------------------------
class ResidualPoseDataset(Dataset):
    def __init__(self, file_path, pose_key="ftip_pose"):
        """
        data: list of dicts, each with
          q : (n_dof,)
          p_fk : (3,)
          R_fk : (3,3)
          p_gt : (3,)
          R_gt : (3,3)
          residual : (6,) = [Δp, δr]
        """
        self.dataset = pickle.load(open(file_path, "rb"))
        self.pose_key = pose_key

    def __len__(self):
        return len(self.dataset['t'])

    def __getitem__(self, idx):
        q = torch.from_numpy(np.concatenate((self.dataset['ur_js'][idx], self.dataset['inspire_js'][idx]))).float()
        p_fk = torch.from_numpy(self.dataset[self.pose_key]["pred"][idx][:3, 3]).float()
        R_fk = torch.from_numpy(self.dataset[self.pose_key]["pred"][idx][:3, :3]).float()
        p_gt = torch.from_numpy(self.dataset[self.pose_key]["gt"][idx][:3, 3]).float()
        R_gt = torch.from_numpy(self.dataset[self.pose_key]["gt"][idx][:3, :3]).float()
        return q, p_fk, R_fk, p_gt, R_gt


# -----------------------------
# Neural Net
# -----------------------------
class MLP(nn.Module):
    def __init__(self, dof, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dof, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)   # Δp (3) + δr (3)
        )

    def forward(self, q):
        return self.mlp(q)
    
class ResidualNet(nn.Module):
    def __init__(self, dof, hidden_dim=256, num_layers=6, dropout=0.1):
        """
        dof: number of joint angles
        hidden_dim: width of hidden layers
        num_layers: number of hidden layers
        dropout: dropout probability for regularization
        """
        super().__init__()

        layers = []
        in_dim = dof
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 6))  # Δp (3) + δr (3)
        self.mlp = nn.Sequential(*layers)

    def forward(self, q):
        return self.mlp(q)


# -----------------------------
# Training loop skeleton
# -----------------------------
def train(model, trainloader, testloader, epochs=50, lr=1e-3, w_rot=1.0):
    min_test_loss = np.inf
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for q, p_fk, R_fk, p_gt, R_gt in trainloader:
            # Predict residual
            pred_residual = model(q)  # (B,6)
            dp, dr = pred_residual[:, :3], pred_residual[:, 3:]

            # Correct pose
            p_pred = p_fk + dp
            R_delta = axis_angle_to_matrix(dr)
            R_pred = torch.bmm(R_delta, R_fk)

            # Loss
            pos_loss = ((p_pred - p_gt) ** 2).sum(dim=-1).mean()
            rot_loss = rotation_geodesic_loss(R_pred, R_gt).mean()
            loss = pos_loss + w_rot * rot_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss/len(trainloader):.6f}")

        # test at each epoch
        with torch.no_grad():
            test_loss = 0
            for q, p_fk, R_fk, p_gt, R_gt in testloader:
                pred_residual = model(q)
                dp, dr = pred_residual[:, :3], pred_residual[:, 3:]
                p_pred = p_fk + dp
                R_delta = axis_angle_to_matrix(dr)
                R_pred = torch.bmm(R_delta, R_fk)

                pos_loss = ((p_pred - p_gt) ** 2).sum(dim=-1).mean()
                rot_loss = rotation_geodesic_loss(R_pred, R_gt).mean()
                loss = pos_loss + w_rot * rot_loss
                test_loss += loss.item()

            # save checkpoint
            if test_loss/len(testloader) < min_test_loss:
                min_test_loss = test_loss / len(testloader)
                model_path = "../data/residual_model.pth"
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model to {model_path}")

            print(f"Test Loss = {test_loss/len(testloader):.6f}")

def inference(model, dataloader):
    pose_pred = torch.zeros((0, 4, 4))
    for q, p_fk, R_fk, _, _ in dataloader:
        with torch.no_grad():
            # Predict residual
            pred_residual = model(q)  # (B,6)
            dp, dr = pred_residual[:, :3], pred_residual[:, 3:]

            # Correct pose
            p_pred = p_fk + dp
            R_delta = axis_angle_to_matrix(dr)
            R_pred = torch.bmm(R_delta, R_fk)

            batch_size = q.shape[0]
            pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            pose[:, :3, 3] = p_pred
            pose[:, :3, :3] = R_pred

            pose_pred = torch.cat((pose_pred, pose))

    return pose_pred.numpy()


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Suppose you have preprocessed dataset with residual labels
    pose_key = 'ftip_pose'
    bag_path = "/home/jyp/hardware/inspire_hand/exp_data/20250820/rosbag2_2025_08_20-14_32_21"
    train_path = os.path.join(bag_path, "parsed", f"train_dataset.pkl")
    test_path = os.path.join(bag_path, "parsed", f"test_dataset.pkl")
    inference_path = os.path.join(bag_path, "parsed", f"full_dataset.pkl")

    trainset = ResidualPoseDataset(file_path=train_path, pose_key=pose_key)
    testset = ResidualPoseDataset(file_path=test_path, pose_key=pose_key)
    inferenceset = ResidualPoseDataset(file_path=inference_path, pose_key=pose_key)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(trainset, batch_size=32, shuffle=False)
    inference_loader = DataLoader(inferenceset, batch_size=512, shuffle=False)

    model = MLP(dof=6+12)
    # model = ResidualNet(dof=6+12, hidden_dim=256, num_layers=6, dropout=0.1)

    model.load_state_dict(torch.load("../data/residual_model.pth"))

    # train(model, trainloader, testloader)
    pose_pred = inference(model, inference_loader)
    dataset = pickle.load(open(inference_path, "rb"))
    dataset[pose_key]['nn_pred'] = pose_pred.tolist()
    with open(inference_path, "wb") as f:
        pickle.dump(dataset, f)
