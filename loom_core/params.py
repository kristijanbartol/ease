import json
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


SUBJECT_BETAS = np.array([[1.09, 1.16, 1.51, 1.58, 2.1, -0.2, 0.2636, -2.3, -2.55, 0.8]], dtype=np.float32)


def zero_shape():
    return torch.zeros((1, 10))

def subject_shape():
    return torch.tensor(SUBJECT_BETAS)

def s_s_shape():
    # NOTE: I updated this to use for the male (!!!)
    shape = torch.zeros((1, 10))
    shape[:, 1:10] = 2.5
    shape[0, 0] = -1.75
    return shape

def male_s_s_shape():
    shape = torch.zeros((1, 10))
    shape[:, 1:10] = 2.0
    shape[0, 0] = -1.75
    return shape

def l_l_shape():
    shape = torch.zeros((1, 10))
    shape[:, 1:10] = -2.5
    #shape[:, 1:10] = -1.75
    shape[0, 0] = 1.75
    #shape[0, 0] = 1.25
    return shape

# special shape parameters for male due to issues
'''
def male_l_l_shape():
    shape = torch.zeros((1, 10))
    shape[:, 1:10] = -1.5
    shape[0, 0] = 1.5
    return shape

def ml_ml_shape():
    shape = torch.zeros((1, 10))
    shape[:, 1:10] = -1.25
    shape[0, 0] = 1.25
    return shape
'''

def t_pose():
    pose = torch.zeros((1, 23 * 3))
    #pose[0, 0*3:1*3] = torch.tensor([0, 0, np.pi / 16])
    #pose[0, 1*3:2*3] = torch.tensor([0, 0, -np.pi / 16])
    return pose

def t_pose_for_cutting():
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

def sit_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 0*3:1*3] = torch.tensor([-np.pi / 2, 0, 0]) # left hip
    pose[0, 1*3:2*3] = torch.tensor([-np.pi / 2, 0, 0]) # right hip
    pose[0, 3*3:4*3] = torch.tensor([np.pi / 2, 0, 0])  # left knee
    pose[0, 4*3:5*3] = torch.tensor([np.pi / 2, 0, 0])  # right knee
    
    return pose

def arms_up_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 12*3:13*3] = torch.tensor([0, 0, np.pi / 4])  # left arm
    pose[0, 13*3:14*3] = torch.tensor([0, 0, -np.pi / 4]) # right arm
    pose[0, 0*3:1*3] = torch.tensor([0, 0, np.pi / 16])
    pose[0, 1*3:2*3] = torch.tensor([0, 0, -np.pi / 16])
    return pose

def standard5_pose():       # arms up + legs scissors pose
    pose = arms_up_pose()
    pose[0, 0*3:1*3] = torch.tensor([np.pi / 6, 0, 0])
    pose[0, 1*3:2*3] = torch.tensor([-np.pi / 6, 0, 0])
    return pose

def arms_front_bent():
    pose = torch.zeros((1, 23 * 3))
    # Arms front
    pose[0, 15*3:16*3] = torch.tensor([0, -np.pi / 2, 0])  # left arm
    pose[0, 16*3:17*3] = torch.tensor([0, np.pi / 2, 0]) # right arm
    # Arms bent
    pose[0, 17*3:18*3] = torch.tensor([0, 0, np.pi / 2])  # left arm
    pose[0, 18*3:19*3] = torch.tensor([0, 0, -np.pi / 2]) # right arm
    return pose

def standard4_pose():       # arms front bent + legs apart
    pose = arms_front_bent()
    pose[0, 0*3:1*3] = torch.tensor([0, 0, np.pi / 8])
    pose[0, 1*3:2*3] = torch.tensor([0, 0, -np.pi / 8])
    return pose

def arms_front_pose():
    pose = torch.zeros((1, 23 * 3))
    pose[0, 15*3:16*3] = torch.tensor([0, -np.pi / 2, 0])  # left arm
    pose[0, 16*3:17*3] = torch.tensor([0, np.pi / 2, 0]) # right arm
    return pose

def standard2_pose():       # arms front + legs apart
    pose = arms_front_pose()
    pose[0, 0*3:1*3] = torch.tensor([0, 0, np.pi / 8])
    pose[0, 1*3:2*3] = torch.tensor([0, 0, -np.pi / 8])
    return pose


REF_KPTS = {
    'upper': {
        "head": [444, 414],
        'mid': [3168, 3500],
        'neck': [4294, 5310],
        'shoulder': [5282, 5335],
        'side': [5326, 4891],

        'sleeve': None,
        'bottom': None
    },
    'lower': {
        'side': [4164, 4303],
        #'side': [6374, 6374],
        'between': [1208, 1364],

        'bottom_side_ref': 6604,    # references for the direction of the cut
        #'bottom_side_ref': 6728,
        #'bottom_side_ref': 6729,
        'bottom_inner_ref': 6833
        #'bottom_inner_ref': 6610
    }
}

def map_dict(src_V, dst_V, ref_dict, tol=1e-8):
    tree = cKDTree(dst_V)
    def map_val(v):
        if v is None: return None
        if isinstance(v, list):
            d, j = tree.query(src_V[v], k=1)
            return j.tolist()
        else:
            d, j = tree.query(src_V[v], k=1)
            return int(j)
    return {k:{kk:map_val(vv) for kk,vv in d.items()} for k,d in ref_dict.items()}

REF_KPTS_SKIRTIFIED = {
    'upper': {
        "head": [444, 414],
        'mid': [3149, 3479],
        'neck': [4273, 5232],
        'shoulder': [5204, 5257],
        'side': [5248, 4830],

        'sleeve': None,
        'bottom': None
    },
    'lower': {
        'side': [4143, 4282],

        'bottom_side_ref': 6520    # references for the direction of the cut
    }
}

### DSL LOGIC ###
# 1. Flags fully determine the type and structure of the design
# 2. Reference keypoints define the high-level template
# 3. Keypoints, flags, and position, and length parameters fully define the design
# 3. Fully custom designs directly specify all the keypoints


DESIGN_TEMPLATE = {
    'upper': {
        'pos': {
            'head': 0.5,
            'mid': 0.1,
            'neck': 0.1,      # assym.
            'shoulder': 0.7,  # assym.
            'side': 0.3,      # assym.
        },
        'length': {
            'sleeve': 0.25,
            'bottom': 0.25,   # assym.
        },
        'flag': {
            'use_head': False,
            'is_shoulderless': False,
            'use_mid': False,
            'use_sleeve': True,
            'is_dress': False,
            'is_assymetric': False
        }
    },
    'lower': {
        'pos': {
            'side': 0.5,
            'between': 0.7
        },
        'length': {
            'bottom': 0.7
        },
        'flag': {
            'is_skirt': False
        }
    }
}


HYPERPARAMS_TEMPLATE = {
    "stretch_coef": 2.0,
    "edges_coef": 1.0,
    "seams_coef": 0.0,
    "material_stretch_coef": 1.0,
    "seamline_strategy": "average",
    "matching_mode": "strict",
    "num_seam_iters": 1,
    "num_inner_iters": 1,
    "max_stretch": 0.05,
    "dart_coef": 50.0,
    "equalize_seamline_lengths": False
}


def process_config(config):
    experiment_name = f'{config["body_set"]}_{config["upper_design_label"]}-{config["lower_design_label"]}'
    experiment_name += f'-{config["upper_scales"]}-{config["lower_scales"]}-{config["upper_max_scale"]}-{config["lower_max_scale"]}'
    
    design_params = {}
    with open(f'config/designs/upper/{config["upper_design_label"]}.json') as upper_f:
        design_params['upper'] = json.load(upper_f)
    with open(f'config/designs/lower/{config["lower_design_label"]}.json') as lower_f:
        design_params['lower'] = json.load(lower_f)
    with open(f'config/hyperparams/{config["hyperparams_label"]}.json') as hyper_f:
        hyperparams = json.load(hyper_f)
    with open(f'config/body_sets/{config["body_set"]}.json') as body_set_f:
        body_set = json.load(body_set_f)
        body_set['name'] = config['body_set']

    for garment_part in ['upper', 'lower']:
        design_params[garment_part]['scales'] = config[f'{garment_part}_scales']
        max_scale = config[f'{garment_part}_max_scale']
        if max_scale > 0:
            design_params[garment_part]['max_scale'] = max_scale
        else:
            design_params[garment_part]['max_scale'] = None

    return experiment_name, design_params, hyperparams, body_set


# Q: Why wouldn't I be able to simply define the keypoints in a row and cut the proper patches based on these
# Specific value definition: for each new set of values (different than before), create a new design within the corresponding folder with JSON.
#       NOTE: when the design with the same values is created, rename it with the newest index (so that you can get consecutive indices for ranges).
PARAM_DESIGNS = {

    'upper': {
        'base': {
            'neck': 0.1,
            'shoulder': 0.7,
            'side': 0.3,

            'sleeve': 0.25,
            'bottom': 0.25
        },

        'base_sleeveless': {
            'neck': 0.1,
            'shoulder': 0.7,
            'side': 0.3,

            'bottom': 0.25
        },

        'base_shoulderless': {
            'mid': 0.3,     # TODO: remove the need for this keypoint - when no keypoint in the middle, connect automatically
            'side': 0.3,

            'bottom': 0.25
        },

        'decolte': {
            'mid': 0.7,
            'neck': 0.1,
            'shoulder': 0.7,
            'side': 0.3,

            'sleeve': 0.25,
            'bottom': 0.25
        },

        'decolte_sleeveless': {
            'mid': 0.7,
            'neck': 0.1,
            'shoulder': 0.7,
            'side': 0.3,

            'bottom': 0.25
        },

        'decolte_shoulderless': {
            'mid': 0.3,     # TODO: remove the need for this keypoint - when no keypoint in the middle, connect automatically
            'side': 0.3,

            'bottom': 0.25
        },

        'dress_base': {
            'neck': 0.1,
            'shoulder': 0.7,
            'side': 0.3,

            'sleeves': 0.1,
            'bottom': 0.8
        },

        'dress_sleeveless': {
            'neck': 0.1,
            'shoulder': 0.7,
            'side': 0.1,

            'bottom': 0.8
        },

        'dress_shoulderless': {
            'mid': 0.3,     # TODO: remove the need for this keypoint - when no keypoint in the middle, connect automatically
            'side': 0.3,

            'bottom': 0.8
        }
    },

    # LOWER #

    'lower': {
        'base': {
            'side': 0.5,
            'between': 0.7,

            'bottom': 0.7
        }
    }

    ##########
}

params1 = {
    'upper': {
        'mid': 0.7,         # [0., 1.]
        'neck': 0.1,        # [0., 1.]
        'shoulder': 0.7,    # [0., 1.]
        'side': 0.3,      # [0., 1.]

        'sleeve': 0.25,      # [m]
        'bottom': 0.25      # [m]
    },
    'lower': {
        'side': 0.5,         # [0., 1.]
        'between': 0.7,     # [0., 1.]

        'bottom': 0.7       # [m]
    }
}


if __name__ == '__main__':
    # This block is used only "once" - for generating skirtified keypoints.
    orig_mesh = trimesh.load('data/meshes/ref.ply')
    skirtified_mesh = trimesh.load('data/skirtified/solo-female/ref.ply')

    skirtified_kpts = map_dict(
        src_V=orig_mesh.vertices,
        dst_V=skirtified_mesh.vertices,
        ref_dict=REF_KPTS
    )

    print(skirtified_kpts)
