from smplx import SMPL
import argparse
import torch
import numpy as np

from const import (
    PANT_LENGTH,
    SHIRT_LENGTH,
    SLEEVE_LENGTH
)
from selector import (
    select_original,
    select_subdivided
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gender', '-G', type=str, choices=['male', 'female', 'neutral'], default='female')
    parser.add_argument('--file_format', '-F', type=str, choices=['ply', 'obj', 'both'], default='ply')
    parser.add_argument('--subdivide', dest='subdivide', action='store_true')
    parser.add_argument('--shirt_length', '-S', type=float, default=SHIRT_LENGTH)
    parser.add_argument('--pant_length', '-P', type=float, default=PANT_LENGTH)
    parser.add_argument('--sleeve_length', '-L', type=float, default=SLEEVE_LENGTH)
    args = parser.parse_args()

    '''
    pose = torch.zeros((1, 23 * 3))
    # 0, 1 -> left leg
    # 1, 2 -> right leg
    # 2, 3 -> mid-hip
    # 3, 4 -> left knee
    i, j = 3, 4
    pose[0, i*3:j*3] = torch.tensor([np.pi / 2 / 2, 0, 0])
    '''

    smpl_model = SMPL(model_path='/home/kristijan/data/hierprob3d/smpl/SMPL_FEMALE.pkl',
                      #body_pose=pose)
    )

    if args.subdivide:
        select_subdivided(args, smpl_model)
    else:
        select_original(args, smpl_model)
