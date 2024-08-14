import numpy as np
from smplx import SMPL
from collections import defaultdict
import torch
import trimesh


SEGMENTS_DIR = 'config/segments/'

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

MID_LINE_FRONT = [3060, 3168, 3169, 3171, 3078, 3073, 3074, 3495, 3496, 3497, 3498, 3506, 3076, 3077, 3079, 1329, 1330, 3511,
                  1325, 1326, 3509, 3504, 3501, 3500, 1768, 1769, 3503, 3507, 3160, 1806, 1807, 3510, 3145, 3146]
MID_LINE_BACK = [3470, 3012, 1306, 1305, 2878, 2877, 3471, 1755, 1754, 3482, 3029, 3028, 3027, 3508, 3015, 3014, 3505, 3017,
                 3016, 3173, 3024, 3023, 3502, 3022, 3021, 1784, 1783, 3159, 3158, 3484, 3120]
RIGHT_ARMPIT = [
    4767, 4766, 4892, 4891, 6412, 4132, 4131, 4134, 4106, 4109, 6300, 4959, 4962,
    4804, 4118, 4121, 4165, 4332, 6378, 5259, 5258, 4423, 4311, 4927, 4801, 4712,
    4334, 4335, 4390, 4459, 4460, 4466, 4465, 4570, 4519, 4496, 4495, 4539, 4556,
    4559
]
RIGHT_ARMPIT2 = [
    4768, 4765, 4228, 4227, 6413, 4133, 4130, 4135, 4107, 4108, 6301, 4960, 4961, 4963, 4119, 4120, 4164,
    4317, 6375, 5257, 5256, 4303, 4298, 4297, 4920, 6550, 4745, 4430, 4389
]
LEFT_ARMPIT = [
    1285, 1286, 1418, 1419, 2953, 644, 645, 646, 618, 619, 2839, 1487, 1488, 1323, 
    630, 631, 677, 846, 2919, 1795, 1796, 937, 822, 823, 1454, 1321, 1228, 850, 849, 
    905, 973, 974, 980, 979, 1022, 1033, 1009, 1010, 1052, 1070, 1072
]
LEFT_ARMPIT2 = [
    1284, 1283, 740, 741, 2954, 643, 642, 647, 621, 620, 2840, 1490, 1489, 1492, 633,
    632, 676, 830, 2915, 1793, 1794, 816, 808, 809, 1446, 3129
]
LEFT_ARMPIT_SKIRTIFIED = [1278, 1279, 1407, 1408, 2935, 644, 645, 646, 618, 619,
                          2821, 1474, 1475, 1316, 630, 631, 677, 846, 2901, 1777, 1778,
                          936, 822, 823, 1441, 1314, 1223, 850, 849, 904, 972, 973, 979,
                          978, 1021, 1032, 1008, 1009, 1051, 1069, 1071, 1100, 1099, 1457,
                          1117, 1116, 1135, 3168, 3169]
RIGHT_ARMPIT_SKIRTIFIED = [4710, 4709, 4831, 4830, 6334, 4111, 4110, 4113, 4085,
                           4088, 6222, 4894, 4897, 4747, 4097, 4100, 4144, 4311, 6300, 5181,
                           5180, 4401, 4290, 4289, 4863, 4744, 4656, 4313, 4314, 4368, 4428,
                           4429, 4435, 4434, 4472, 4482, 4465, 4464, 4500, 4514, 4517, 4540,
                           4541, 4878, 4556, 4557, 4574, 6507, 6508]
RIGHT_SHOULDER = [6469, 5322, 5325, 4721, 4724, 4270, 4198, 4094, 4097, 4230, 4306, 4305]
RIGHT_SHOULDER2 = [6470, 5342, 5285, 5282, 5310, 4269, 4271, 5272, 4232, 4231, 4294, 4291]
RIGHT_SHOULDER_SKIRTIFIED = [6392, 5264, 5207, 5204, 5232, 4248, 4250, 5194, 4211]
LEFT_SHOULDER = [3010, 1861, 1862, 1238, 1239, 783, 711, 606, 607, 742, 818, 817]
LEFT_SHOULDER2 = [3011, 1881, 1822, 1821, 1849, 781, 782, 1810, 743, 744, 804, 803]
LEFT_SHOULDER_SKIRTIFIED = [2993, 1863, 1804, 1803, 1831, 781, 782, 1792, 743]
BOTTOM_SHIRT = [1807, 864, 863, 1205, 1204, 1450, 1799, 868, ]
RIGHT_OUTER_PANT = [6378, 5259, 5258, 4423, 4311, 4310, 4927, 4801, 4712, 4334, 4335, 4390, 4459, 4460, 
                    4466, 4465, 4507, 4519, 4496, 4495, 4539, 4556, 4559, 4585, 4586, 4943, 
                    4603, 4604, 4622, 6589, 6590, 6608, 6869]
LEFT_OUTER_PANT = [2919, 1795, 1796, 937, 822, 823, 1454, 1321, 1228, 850, 849, 905, 973, 974, 980, 979, 
                   1022, 1033, 1009, 1010, 1052, 1070, 1072, 1101, 1100, 1470, 1118, 1117, 
                   1136, 3189, 3190, 3208, 3469]
RIGHT_INNER_PANT = [1208, 4364, 4367, 4925, 6553, 4709, 4708, 4450, 4449, 4435, 4438, 4439, 
                    4442, 4514, 4527, 4502, 4501, 4543, 4542, 4565, 4996, 4574, 4573, 4591, 
                    4590, 4609, 6577, 6576, 6598, 6833]
RIGHT_INNER_PANT2 = [1209, 4650, 4651, 4970, 6556, 4948, 4834, 4835, 4926, 4928, 4929, 4988, 4989,
                     4990, 4991, 4992, 4993, 4994, 4995, 4998, 4997, 4999, 5000, 5001, 5002, 5003,
                     6596, 6597, 6610]
LEFT_INNER_PANT = [1208, 878, 879, 1451, 3131, 1226, 1227, 963, 964, 949, 950, 953, 954, 1028, 
                   1042, 1014, 1015, 1059, 1056, 1078, 1525, 1088, 1089, 1107, 1104, 1122, 
                   3175, 3176, 3198, 3433]
LEFT_INNER_PANT2 = [1209, 1166, 1165, 1498, 3135, 1475, 1360, 1359, 1453, 1456, 1455, 1518, 1517,
                   1519, 1520, 1521, 1522, 1523, 1524, 1526, 1527, 1528, 1530, 1529, 1531, 1532,
                   3197, 3196, 3210]
FRONT_INNER_PANT = [3507, 3160, 1806, 1807, 3510, 3145, 3146, 3148, 3149, 1208]
FRONT_INNER_PANT2 = [3507, 3160, 1806, 1807, 3510, 3145, 3146, 3148, 3149, 1208, 1209]
BACK_INNER_PANT = [1784, 1783, 3159, 3158, 3484, 3120, 3119, 3141, 3481, 3102, 1540, 1539, 1476, 1364, 1363, 1515, 3172, 3170, 1353, 1278, 1210, 1209, 1208]
BACK_INNER_PANT2 = [1784, 1783, 3159, 3158, 3484, 3120, 3119, 3141, 3481, 3102, 1540, 1539, 1476, 1364, 1363, 1515, 3172, 3170, 1353, 1278, 1210, 1209, 1208, 1209]
RIGHT_FRONT_ARM = [4767, 5008, 4685, 4163, 4162, 5359, 6467, 6468, 6469]
RIGHT_FRONT_ARM_SKIRTIFIED = [6392, 6391, 6390, 6389, 5281, 4141, 4142, 4632, 4930, 4710]
LEFT_FRONT_ARM = [1285, 1537, 1199, 673, 674, 1898, 3008, 3009, 3010]
LEFT_FRONT_ARM_SKIRTIFIED = [2993, 2992, 2991, 2990, 1880, 674, 673, 1196, 1519, 1278]
RIGHT_BACK_ARM = [6470, 6352, 6351, 5346, 5345, 5352, 6462, 6459, 6402, 5302, 4200, 4203, 5340, 4243, 4242, 4769, 4768, 4767, 6469]
RIGHT_BACK_ARM_SKIRTIFIED = [4710, 4711, 4712, 4221, 4222, 5262, 4182, 4179, 5224, 6324, 6381, 6384, 5274, 5267, 5268, 6273, 6274, 6392]
LEFT_BACK_ARM = [3011, 2893, 2894, 1887, 1884, 1891, 3003, 3000, 2943, 1841, 712, 713, 1879, 755, 756, 1288, 1284, 1285]
LEFT_BACK_ARM_SKIRTIFIED = [1278, 1277, 1281, 756, 755, 1861, 713, 712, 1823, 2925, 2982, 2985, 1873, 1866, 1869, 2876, 2875, 2993]
LEFT_SLEEVE_UP = [3010, 636, 635, 1860, 1400, 1399, 1407, 1406, 1311, 1310, 1679, 1639, 1640, 1653, 1654, 1613, 1614, 1915, 
                  1742, 1691, 1692, 1557, 1556, 1951, 1950, 1925, 1926, 2109, 2019]
LEFT_SLEEVE_DOWN = [1285, 682, 681, 1866, 1282, 1281, 791, 790, 1385, 1386, 1671, 1631, 1632, 1733, 1662, 1627, 1628, 1917, 
                      2873, 1722, 1721, 1689, 1690, 1978, 1942, 1941, 1979, 2108, 2060]
RIGHT_SLEEVE_UP = [6469, 4124, 4125, 5321, 4872, 4873, 4880, 4881, 4791, 4790, 5148, 5108, 5111, 5122, 5125, 5082, 5085, 
                  5376, 5211, 5161, 5160, 5028, 5025, 5412, 5413, 5387, 5386, 5570, 5480]
RIGHT_SLEEVE_DOWN = [4767, 4170, 4171, 5326, 4763, 4764, 4281, 4278, 4859, 4858, 5140, 5100, 5103, 5201, 5130, 5096, 5099,
                       5378, 6334, 5191, 5190, 5159, 5158, 5438, 5401, 5402, 5440, 5568]
FRONT_NECK_SKIRTIFIED = [4211, 4076, 4075, 4722, 4723, 4724, 4725, 6424, 3151, 3049, 1293, 1292, 1291, 1290, 608, 607, 743]
BACK_NECK_SKIRTIFIED = [743, 1793, 1800, 1801, 1802, 2857, 2858, 2860, 6258, 6259, 5202, 5203, 5201, 5192, 4211]

# Use alternative seam vertices
#LEFT_SHOULDER = LEFT_SHOULDER2
#RIGHT_SHOULDER = RIGHT_SHOULDER2
LEFT_ARMPIT = LEFT_ARMPIT2
RIGHT_ARMPIT = RIGHT_ARMPIT2
#LEFT_INNER_PANT = LEFT_INNER_PANT2
#RIGHT_INNER_PANT = RIGHT_INNER_PANT2
#FRONT_INNER_PANT = FRONT_INNER_PANT2
#BACK_INNER_PANT = BACK_INNER_PANT2

SEGMENT_SETS = {
    'default': [
        'upper_front',
        'upper_back',
        'sleeve_front_right',
        'sleeve_back_right',
        'sleeve_front_left',
        'sleeve_back_left',
        'lower_front_right',
        'lower_back_right',
        'lower_front_left',
        'lower_back_left'
    ],
    'half': [
        'upper_front_right',
        'upper_front_left',
        'upper_back_right',
        'upper_back_left',
        'sleeve_front_right',
        'sleeve_back_right',
        'sleeve_front_left',
        'sleeve_back_left',
        'lower_front_right',
        'lower_back_right',
        'lower_front_left',
        'lower_back_left'
    ]
}

THRESH_TO_SEAMS_DICT = {
    'upper': 'left_armpit',
    'lower': 'left_outer',
    'sleeve_left': 'up',
    'sleeve_right': 'up'
}

THRESH_TO_SEGMENT_DICT = {
    'upper': 'upper_front',
    'lower': 'lower_front_left',
    'sleeve_left': 'sleeve_front_left',
    'sleeve_right': 'sleeve_front_right'
}

PLANE_ORIENT_DICT = {
    'upper': 'horizontal',
    'lower': 'horizontal',
    'sleeve_left': 'vertical',
    'sleeve_right': 'vertical'
}

SEGMENT_TO_THRESH_DICT = {
    'upper_front_right': 'upper',
    'upper_front_left': 'upper',
    'upper_back_right': 'upper',
    'upper_back_left': 'upper',
    'sleeve_front_right': 'sleeve_right',
    'sleeve_back_right': 'sleeve_right',
    'sleeve_front_left': 'sleeve_left',
    'sleeve_back_left': 'sleeve_left',
    'lower_front_right': 'lower',
    'lower_back_right': 'lower',
    'lower_front_left': 'lower',
    'lower_back_left': 'lower'
}

# TODO: Rename to SEGMENT_TO_SEAM_IDX_DICT
SEAM_IDX_DICT = {
    'default': {
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
            'neck':         [817, 803, 806, 813, 812, 1219, 3470, 4702, 4302, 4301, 4292, 4291, 4305],
            'right_shoulder': RIGHT_SHOULDER,
            'left_shoulder':  LEFT_SHOULDER
        },
        'sleeve_front_right': {
            'up': RIGHT_SLEEVE_UP,
            'down': RIGHT_SLEEVE_DOWN,
            'side': RIGHT_FRONT_ARM
        },
        'sleeve_back_right': {
            'up': RIGHT_SLEEVE_UP,
            'down': RIGHT_SLEEVE_DOWN,
            'side': RIGHT_BACK_ARM
        },
        'sleeve_front_left': {
            'up':  LEFT_SLEEVE_UP,
            'down': LEFT_SLEEVE_DOWN,
            'side': LEFT_FRONT_ARM
        },
        'sleeve_back_left': {
            'up':  LEFT_SLEEVE_UP,
            'down': LEFT_SLEEVE_DOWN,
            'side': LEFT_BACK_ARM
        },
        'lower_front_right': {
            'right_outer': RIGHT_OUTER_PANT,
            'right_inner': RIGHT_INNER_PANT,
            'mid_inner': FRONT_INNER_PANT,
            'waistline': [6378, 6383, 6385, 6386, 6379, 6380, 6384, 6381, 6382, 3507]
        },
        'lower_front_left': {
            'left_outer' : LEFT_OUTER_PANT,
            'left_inner' : LEFT_INNER_PANT,
            'mid_inner': FRONT_INNER_PANT,
            'waistline': [3507, 2922, 2923, 2925, 2920, 2921, 2926, 2927, 2924, 2919]
        },
        'lower_back_right': {
            'right_outer': RIGHT_OUTER_PANT,
            'right_inner': RIGHT_INNER_PANT,
            'mid_inner': BACK_INNER_PANT,
            'waistline': [1784, 5246, 5247, 6544, 6371, 6370, 6376, 6377, 6374, 6375, 6378]
        },
        'lower_back_left': {
            'left_outer' : LEFT_OUTER_PANT,
            'left_inner' : LEFT_INNER_PANT,
            'mid_inner': BACK_INNER_PANT,
            'waistline': [2919, 2915, 2916, 2917, 2918, 2911, 2910, 3122, 1780, 1781, 1784]
        }
    },
    'dress': {
        'upper_front': {
            'right_armpit': RIGHT_ARMPIT_SKIRTIFIED, 
            'left_armpit':  LEFT_ARMPIT_SKIRTIFIED,
            'left_arm': LEFT_FRONT_ARM_SKIRTIFIED,
            'right_arm': RIGHT_FRONT_ARM_SKIRTIFIED,
            'neck': FRONT_NECK_SKIRTIFIED,
            'right_shoulder': RIGHT_SHOULDER_SKIRTIFIED,
            'left_shoulder':  LEFT_SHOULDER_SKIRTIFIED
        },
        'upper_back': {
            'right_armpit': RIGHT_ARMPIT_SKIRTIFIED, 
            'left_armpit':  LEFT_ARMPIT_SKIRTIFIED,
            'left_arm': LEFT_BACK_ARM_SKIRTIFIED,
            'right_arm': RIGHT_BACK_ARM_SKIRTIFIED,
            'neck': BACK_NECK_SKIRTIFIED,
            'right_shoulder': RIGHT_SHOULDER_SKIRTIFIED,
            'left_shoulder':  LEFT_SHOULDER_SKIRTIFIED
        }
    }
}

SEGMENT_TO_ID = {
    'default': {
        'upper_front': 0,
        'upper_back': 1,
        'sleeve_front_left': 2,
        'sleeve_back_left': 3,
        'sleeve_front_right': 4,
        'sleeve_back_right': 5,
        'lower_front_left': 6,
        'lower_back_left': 7,
        'lower_front_right': 8,
        'lower_back_right': 9
    },
    'dress': {
        'upper_front': 0,
        'upper_back': 1
    }
}

ID_TO_SEGMENT = {
    'default': {
        0: 'upper_front',
        1: 'upper_back',
        2: 'sleeve_front_left',
        3: 'sleeve_back_left',
        4: 'sleeve_front_right',
        5: 'sleeve_back_right',
        6: 'lower_front_left',
        7: 'lower_back_left',
        8: 'lower_front_right',
        9: 'lower_back_right'
    },
    'dress': {
        0: 'upper_front',
        1: 'upper_back'
    }
}

# More compact version of the above two dictionaries
#SEGMENT_TO_ID = {key: idx for idx, key in enumerate(SEAM_IDX_DICT.keys())}
#ID_TO_SEGMENT = {idx: key for idx, key in enumerate(SEAM_IDX_DICT.keys())}

SEAM_TO_SEAM_IDX_DICT = {
    'default': {
        'left_armpit': LEFT_ARMPIT,
        'left_front_arm': LEFT_FRONT_ARM,
        'left_shoulder': LEFT_SHOULDER,
        'left_back_arm': LEFT_BACK_ARM,
        'right_armpit': RIGHT_ARMPIT,
        'right_front_arm': RIGHT_FRONT_ARM,
        'right_shoulder': RIGHT_SHOULDER,
        'right_back_arm': RIGHT_BACK_ARM,
        'left_up_arm': LEFT_SLEEVE_UP,
        'left_down_arm': LEFT_SLEEVE_DOWN,
        'right_up_arm': RIGHT_SLEEVE_UP,
        'right_down_arm': RIGHT_SLEEVE_DOWN,
        'left_outer_pant': LEFT_OUTER_PANT,
        'right_outer_pant': RIGHT_OUTER_PANT,
        'left_inner_pant': LEFT_INNER_PANT,
        'right_inner_pant': RIGHT_INNER_PANT,
        'front_inner_pant': FRONT_INNER_PANT,
        'back_inner_pant': BACK_INNER_PANT
    },
    'dress': {
        'left_armpit': LEFT_ARMPIT_SKIRTIFIED,
        'left_shoulder': LEFT_SHOULDER_SKIRTIFIED,
        'right_armpit': RIGHT_ARMPIT_SKIRTIFIED,
        'right_shoulder': RIGHT_SHOULDER_SKIRTIFIED
    }
}

SEAM_TO_SEGMENT_PAIRS = {
    'default': {
        'left_armpit': (0, 1),
        'left_front_arm': (0, 2),
        'left_shoulder': (0, 1),
        'left_back_arm': (1, 3),
        'right_armpit': (0, 1),
        'right_front_arm': (0, 4),
        'right_shoulder': (0, 1),
        'right_back_arm': (1, 5),
        'left_up_arm': (2, 3),
        'left_down_arm': (2, 3),
        'right_up_arm': (4, 5),
        'right_down_arm': (4, 5),
        'left_outer_pant': (6, 7),
        'right_outer_pant': (8, 9),
        'front_inner_pant': (6, 8),
        'back_inner_pant': (7, 9),
        'left_inner_pant': (6, 7),
        'right_inner_pant': (8, 9)
    },
    'dress': {
        'left_armpit': (0, 1),
        'left_shoulder': (0, 1),
        'right_armpit': (0, 1),
        'right_shoulder': (0, 1)
    }
}

SEGMENT_TO_SEAMLINES_DICT = defaultdict(list)
for skirtification_type in SEAM_TO_SEGMENT_PAIRS:
    SEGMENT_TO_SEAMLINES_DICT[skirtification_type] = defaultdict(list)
    for key, (id1, id2) in SEAM_TO_SEGMENT_PAIRS[skirtification_type].items():
        SEGMENT_TO_SEAMLINES_DICT[skirtification_type][id1].append(key)
        SEGMENT_TO_SEAMLINES_DICT[skirtification_type][id2].append(key)
    SEGMENT_TO_SEAMLINES_DICT[skirtification_type] = {id_: tuple(keys) for id_, keys in SEGMENT_TO_SEAMLINES_DICT[skirtification_type].items()}

INIT_IDXS = {
    'upper_front': 4173,
    'upper_front_left': ...,
    'upper_front_right': ...,
    'upper_back': 4238,
    'upper_back_left': ...,
    'upper_back_right': ...,
    'sleeve_front_left': 1429,
    'sleeve_back_left': 2981,
    'sleeve_front_right': 4816,
    'sleeve_back_right': 6453,
    'lower_front_right': 4952,
    'lower_back_right': 4339,
    'lower_front_left': 1479,
    'lower_back_left': 897
}

DISPLACEMENTS = {
    'skintight': 0.0025,
    'loose': 0.005
}


INIT_UPPER_FRONT = 4173
INIT_UPPER_BACK = 4238
INIT_FRONT_RIGHT_SLEEVE = 4816
INIT_BACK_RIGHT_SLEEVE = 6453
INIT_FRONT_LEFT_SLEEVE = 1429
INIT_BACK_LEFT_SLEEVE = 2981
INIT_RIGHT_FRONT_PANT = 6873
INIT_LEFT_FRONT_PANT = 914
INIT_RIGHT_BACK_PANT = 4371
INIT_LEFT_BACK_PANT = 3087
INIT_UPPER_FRONT_SKIRTIFIED = 3474
INIT_UPPER_BACK_SKIRTIFIED = 3461

SHIRT_LENGTH = 0.35
SLEEVE_LENGTH = 0.3
PANT_LENGTH = 0.8

DISCRETE_STEP = 0.001
YARN_DIST = 0.005

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
    pose[0, 0*3:1*3] = torch.tensor([0, 0, np.pi / 16])
    pose[0, 1*3:2*3] = torch.tensor([0, 0, -np.pi / 16])
    return pose


def arms_up_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 12*3:13*3] = torch.tensor([0, 0, np.pi / 4])  # left arm
    pose[0, 13*3:14*3] = torch.tensor([0, 0, -np.pi / 4]) # right arm
    pose[0, 0*3:1*3] = torch.tensor([0, 0, np.pi / 16])
    pose[0, 1*3:2*3] = torch.tensor([0, 0, -np.pi / 16])
    return pose


def sit_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 0*3:1*3] = torch.tensor([-np.pi / 2, 0, 0]) # left hip
    pose[0, 1*3:2*3] = torch.tensor([-np.pi / 2, 0, 0]) # right hip
    pose[0, 3*3:4*3] = torch.tensor([np.pi / 2, 0, 0])  # left knee
    pose[0, 4*3:5*3] = torch.tensor([np.pi / 2, 0, 0])  # right knee
    #pose[0, 15*3:16*3] = torch.tensor([0, -np.pi / 2, 0])  # left arm
    #pose[0, 16*3:17*3] = torch.tensor([0, np.pi / 2, 0]) # right arm
    return pose


def bent_knee_45_pose():
    pose = a_pose()
    pose[0, 0*3:1*3] = torch.tensor([0, 0, np.pi / 16])
    pose[0, 1*3:2*3] = torch.tensor([0, 0, -np.pi / 16])
    pose[0, 3*3:4*3] = torch.tensor([np.pi / 4, 0, 0])
    return pose


def bent_knee_90_pose():
    pose = a_pose()
    pose[0, 0*3:1*3] = torch.tensor([0, 0, np.pi / 16])
    pose[0, 1*3:2*3] = torch.tensor([0, 0, -np.pi / 16])
    pose[0, 3*3:4*3] = torch.tensor([np.pi / 2, 0, 0])
    return pose


def zero_shape():
    shape = torch.zeros((1, 10))
    return shape


def plump_shape():
    shape = torch.zeros((1, 10))
    shape[:, 1:10] = -2.5
    return shape


def slim_shape():
    shape = torch.zeros((1, 10))
    shape[:, 1:10] = 2.5
    return shape


def large_shape():
    shape = torch.ones((1, 10)) * (-2.5)
    return shape


def small_shape():
    shape = torch.ones((1, 10)) * 2.5
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
