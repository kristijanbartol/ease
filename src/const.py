import numpy as np
from smplx import SMPL
import torch
import trimesh


KEYPOINTS = {
    'upper_front': {
        'neck': [4094, 3171, 607],
        'left_shoulder': [607, 3010],
        'left_arm': [3010, 674, 1285],
        'left_side': [1285, 618],
        'right_shoulder': [4094, 6469],
        'right_arm': [6469, 4162, 4767],
        'right_side': [4767, 4106],
        'bottom': []
    }
}


RIGHT_ARMPIT = [
    4767, 4766, 4892, 4891, 6412, 4132, 4131, 4134, 4106, 4109, 6300, 4959, 4962,
    4804, 4118, 4121, 4165, 4332, 6378, 5259, 5258, 4423, 4311, 4927, 4801, 4712,
    4334, 4335, 4390, 4459, 4460, 4466, 4465, 4570, 4519, 4496, 4495, 4539, 4556,
    4559
]
LEFT_ARMPIT = [
    1285, 1286, 1418, 1419, 2953, 644, 645, 646, 618, 619, 2839, 1487, 1488, 1323, 
    630, 631, 677, 846, 2919, 1795, 1796, 937, 822, 823, 1454, 1321, 1228, 850, 849, 
    905, 973, 974, 980, 979, 1022, 1033, 1009, 1010, 1052, 1070, 1072
]
RIGHT_SHOULDER = [6469, 5322, 5325, 4721, 4724, 4270, 4198, 4094, 4097, 4230, 4306, 4305]
LEFT_SHOULDER = [3010, 1861, 1862, 1238, 1239, 783, 711, 606, 607, 742, 818, 817]
RIGHT_OUTER_PANT = [6378, 5259, 5258, 4423, 4311, 4310, 4927, 4801, 4712, 4334, 4335, 4390, 4459, 4460, 
                    4466, 4465, 4507, 4519, 4496, 4495, 4539, 4556, 4559, 4585, 4586, 4943, 
                    4603, 4604, 4622, 6589, 6590, 6608, 6869]
LEFT_OUTER_PANT = [2919, 1795, 1796, 937, 822, 823, 1454, 1321, 1228, 850, 849, 905, 973, 974, 980, 979, 
                   1022, 1033, 1009, 1010, 1052, 1070, 1072, 1101, 1100, 1470, 1118, 1117, 
                   1136, 3189, 3190, 3208, 3469]
RIGHT_INNER_PANT = [1208, 4364, 4367, 4925, 6553, 4709, 4708, 4450, 4449, 4435, 4438, 4439, 
                    4442, 4514, 4527, 4502, 4501, 4543, 4542, 4565, 4996, 4574, 4573, 4591, 
                    4590, 4609, 6577, 6576, 6598, 6833]
LEFT_INNER_PANT = [1208, 878, 879, 1451, 3131, 1226, 1227, 963, 964, 949, 950, 953, 954, 1028, 
                   1042, 1014, 1015, 1059, 1056, 1078, 1525, 1088, 1089, 1107, 1104, 1122, 
                   3175, 3176, 3198, 3433]
RIGHT_FRONT_ARM = [4767, 5008, 4685, 4163, 4162, 5359, 6467, 6468, 6469]
LEFT_FRONT_ARM = [1285, 1537, 1199, 673, 674, 1898, 3008, 3009, 3010]
RIGHT_BACK_ARM = [6470, 6352, 6351, 5346, 5345, 5352, 6462, 6459, 6402, 5302, 4200, 4203, 5340, 4243, 4242, 4769, 4768, 4767, 6469]
LEFT_BACK_ARM = [3011, 2893, 2894, 1887, 1884, 1891, 3003, 3000, 2943, 1841, 712, 713, 1879, 755, 756, 1288, 1284, 1285]

SEAM_IDX_DICT = {
    'upper_front': {
        'right_armpit': RIGHT_ARMPIT, 
        'left_armpit':  LEFT_ARMPIT,
        'right_arm':    RIGHT_FRONT_ARM,
        'left_arm':     LEFT_FRONT_ARM,
        'neck':         [4305, 4308, 4309, 4787, 4788, 4058, 4059, 3060, 573, 570, 1308, 1307, 821, 819, 817],
        'right_shoulder': RIGHT_SHOULDER,
        'left_shoulder':  LEFT_SHOULDER
    },
    'upper_back': {
        'right_armpit': RIGHT_ARMPIT, 
        'left_armpit':  LEFT_ARMPIT,
        'right_arm':    RIGHT_BACK_ARM,
        'left_arm':     LEFT_BACK_ARM,
        'neck':         [817, 803, 806, 813, 812, 1219, 3470, 3470, 4702, 4302, 4301, 4292, 4291, 4305],
        'right_shoulder': RIGHT_SHOULDER,
        'left_shoulder':  LEFT_SHOULDER
    },
    'sleeve_front_right': {
        'up': [6469, 4124, 4125, 5321, 4872, 4873, 4880, 4881, 4791, 4790, 5148, 5108, 5111, 5122, 5125, 5082, 5085, 
                  5376, 5211, 5161, 5160, 5028, 5025, 5412, 5413, 5387, 5386, 5570, 5480],
        'down': [4767, 4170, 4171, 5326, 4763, 4764, 4281, 4278, 4859, 4858, 5140, 5100, 5103, 5201, 5130, 5096, 5099,
                       5378, 6334, 5191, 5190, 5159, 5158, 5438, 5401, 5402, 5440, 5568],
        'side': RIGHT_FRONT_ARM
    },
    'sleeve_back_right': {
        'up': [6469, 4124, 4125, 5321, 4872, 4873, 4880, 4881, 4791, 4790, 5148, 5108, 5111, 5122, 5125, 5082, 5085, 
                  5376, 5211, 5161, 5160, 5028, 5025, 5412, 5413, 5387, 5386, 5570, 5480],
        'down': [4767, 4170, 4171, 5326, 4763, 4764, 4281, 4278, 4859, 4858, 5140, 5100, 5103, 5201, 5130, 5096, 5099,
                       5378, 6334, 5191, 5190, 5159, 5158, 5438, 5401, 5402, 5440, 5568],
        'side': RIGHT_BACK_ARM
    },
    'sleeve_front_left': {
        'up':  [3010, 636, 635, 1860, 1400, 1399, 1407, 1406, 1311, 1310, 1679, 1639, 1640, 1653, 1654, 1613, 1614, 1915, 
                  1742, 1691, 1692, 1557, 1556, 1951, 1950, 1925, 1926, 2109, 2019],
        'down': [1285, 682, 681, 1866, 1282, 1281, 791, 790, 1385, 1386, 1671, 1631, 1632, 1733, 1662, 1627, 1628, 1917, 
                      2873, 1722, 1721, 1689, 1690, 1978, 1942, 1941, 1979, 2108, 2060],
        'side': LEFT_FRONT_ARM
    },
    'sleeve_back_left': {
        'up':  [3010, 636, 635, 1860, 1400, 1399, 1407, 1406, 1311, 1310, 1679, 1639, 1640, 1653, 1654, 1613, 1614, 1915, 
                  1742, 1691, 1692, 1557, 1556, 1951, 1950, 1925, 1926, 2109, 2019],
        'down': [1285, 682, 681, 1866, 1282, 1281, 791, 790, 1385, 1386, 1671, 1631, 1632, 1733, 1662, 1627, 1628, 1917, 
                      2873, 1722, 1721, 1689, 1690, 1978, 1942, 1941, 1979, 2108, 2060],
        'side': LEFT_BACK_ARM
    },
    'pant_front_right': {
        'right_outer': RIGHT_OUTER_PANT,
        'right_inner': RIGHT_INNER_PANT,
        'mid_inner': [3507, 3160, 1806, 1807, 3510, 3145, 3146, 3148, 3149, 1208],
        'waistline': [6378, 6383, 6385, 6386, 6379, 6380, 6384, 6381, 6382, 3507]
    },
    'pant_front_left': {
        'left_outer' : LEFT_OUTER_PANT,
        'left_inner' : LEFT_INNER_PANT,
        'mid_inner': [3507, 3160, 1806, 1807, 3510, 3145, 3146, 3148, 3149, 1208],
        'waistline': [3507, 2922, 2923, 2925, 2920, 2921, 2926, 2927, 2924, 2919]
    },
    'pant_back_right': {
        'right_outer': RIGHT_OUTER_PANT,
        'right_inner': RIGHT_INNER_PANT,
        'mid_inner': [1784, 1783, 3159, 3158, 3484, 3120, 3119, 3141, 3481, 3102, 1540, 1539, 1476, 1364, 1363, 1515, 3172, 3170, 1353, 1278, 1210, 1209, 1208],
        'waistline': [1784, 5246, 5247, 6544, 6371, 6370, 6376, 6377, 6374, 6375, 6378]
    },
    'pant_back_left': {
        'left_outer' : LEFT_OUTER_PANT,
        'left_inner' : LEFT_INNER_PANT,
        'mid_inner': [1784, 1783, 3159, 3158, 3484, 3120, 3119, 3141, 3481, 3102, 1540, 1539, 1476, 1364, 1363, 1515, 3172, 3170, 1353, 1278, 1210, 1209, 1208],
        'waistline': [2919, 2915, 2916, 2917, 2918, 2911, 2910, 3122, 1780, 1781, 1784]
    }
}
INIT_UPPER_FRONT = 4173
INIT_UPPER_BACK = 4238
INIT_FRONT_RIGHT_SLEEVE = 4816
INIT_BACK_RIGHT_SLEEVE = 6453
INIT_FRONT_LEFT_SLEEVE = 1429
INIT_BACK_LEFT_SLEEVE = 2981
INIT_RIGHT_FRONT_PANT = 4952
INIT_LEFT_FRONT_PANT = 1479
INIT_RIGHT_BACK_PANT = 4339
INIT_LEFT_BACK_PANT = 897

SHIRT_LENGTH = 0.35
SLEEVE_LENGTH = 0.3
PANT_LENGTH = 0.8

DISCRETE_STEP = 0.001
YARN_DIST = 0.005
DISPLACEMENTS = {
    'skintight': 0.0025,
    'loose': 0.0075
}

GLOBAL_WARP = np.array([0, 0, -1])
GLOBAL_WEFT = np.array([1, 0, 0])

COLOR_MAP = {
    'red': (125, 0, 0),
    'blue': (0, 0, 255),
    'light_green': (128, 255, 128),
    'dark_green': (0, 128, 0),
    'orange': (255, 128, 0),
    'dark_blue': (0, 0, 139),
    'light_blue': (173, 216, 230),
    'brown': (165, 42, 42),
    'yellow': (255, 255, 0),
    'white': (255, 255, 255)
}


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


def t_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 0*3:1*3] = torch.tensor([0, 0, np.pi / 16])
    pose[0, 1*3:2*3] = torch.tensor([0, 0, -np.pi / 16])
    return pose


def a_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 15*3:16*3] = torch.tensor([0, -np.pi / 16, -np.pi / (4 / 1.1)])
    pose[0, 16*3:17*3] = torch.tensor([0, np.pi / 16, np.pi / (4 / 1.1)])
    return pose


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


def bent_knee_45_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 3*3:4*3] = torch.tensor([np.pi / 4, 0, 0])
    return pose


def bent_knee_90_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 3*3:4*3] = torch.tensor([np.pi / 2, 0, 0])
    return pose


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
