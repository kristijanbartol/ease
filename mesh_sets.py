from smplx import SMPL
import torch
import numpy as np
import trimesh


# 0 -> left hip
# 1 -> right hip
# 2 -> mid hip
# 3 -> left knee
# 4 -> right knee
# 5 -> spine-bottom
# 6 -> left ankle
# 7 -> right ankle
# 8 -> spine-top
# 9 -> left foot
# 10 -> right foot
# 11 -> neck
# 12 -> left arm
# 13 -> right arm
# 14 -> head 
# 15 -> left shoulder
# 16 -> right shoulder
# 17 -> left elbow
# 18 -> right elbow
# 19 -> left wrist
# 20 -> right wrist
# 21 -> left hand
# 22 -> right hand


def a_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 15*3:16*3] = torch.tensor([0, -np.pi / 16, -np.pi / (4 / 1.1)])
    pose[0, 16*3:17*3] = torch.tensor([0, np.pi / 16, np.pi / (4 / 1.1)])


def arms_up_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 12*3:13*3] = torch.tensor([0, 0, np.pi / 4])  # left arm
    pose[0, 13*3:14*3] = torch.tensor([0, 0, -np.pi / 4]) # right arm
    return pose


def sit_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 0*3:1*3] = torch.tensor([-np.pi / 2, 0, 0]) # left hip
    pose[0, 1*3:2*3] = torch.tensor([-np.pi / 2, 0, 0]) # right hip
    pose[0, 3*3:4*3] = torch.tensor([np.pi / 2, 0, 0])  # left knee
    pose[0, 4*3:5*3] = torch.tensor([np.pi / 2, 0, 0])  # right knee
    pose[0, 15*3:16*3] = torch.tensor([0, -np.pi / 2, 0])  # left arm
    pose[0, 16*3:17*3] = torch.tensor([0, np.pi / 2, 0]) # right arm
    return pose


def bent_knee_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, i*3:4*3] = torch.tensor([np.pi / 4, 0, 0])


def zero_shape():
    shape = torch.zeros((1, 10))
    return shape


def large_shape():
    shape = torch.zeros((1, 10))
    shape[1:10] = 2.5
    return shape


def small_shape():
    shape = torch.zeros((1, 10))
    shape[1:10] = -2.5
    return shape


SETS = {
    'set1': [
        [a_pose, zero_shape],
        [arms_up_pose, zero_shape],
        [sit_pose, zero_shape],
        [a_pose, large_shape],
        [a_pose, small_shape],
        [arms_up_pose, large_shape],
        [sit_pose, large_shape]
    ],
    'set2': [
        [a_pose, zero_shape],
        [bent_knee_pose, zero_shape]
    ]
} 


def iterate_keypoints():
    poses = torch.zeros((23, 1, 23*3))
    for i in range(23):
        poses[i, 0, i*3:(i+1)*3] = torch.tensor([0, 0, -np.pi / 4])
    return poses


if __name__ == '__main__':
    poses = iterate_keypoints()
    smpl_model = SMPL(model_path='/home/kristijan/data/hierprob3d/smpl/SMPL_FEMALE.pkl')
    for i in range(poses.shape[0]):
        verts = smpl_model(body_pose=poses[i]).vertices[0].cpu().detach().numpy()
        faces = smpl_model.faces
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(f'results/target_meshes/pose_{i}.ply')

    verts = smpl_model(body_pose=arms_up_pose()).vertices[0].cpu().detach().numpy()
    faces = smpl_model.faces
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(f'results/target_meshes/arms_up.ply')

    verts = smpl_model(body_pose=sit_pose()).vertices[0].cpu().detach().numpy()
    faces = smpl_model.faces
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(f'results/target_meshes/sit.ply')
