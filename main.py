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
from mesh_sets import SETS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gender', '-G', type=str, choices=['male', 'female', 'neutral'], default='male')
    parser.add_argument('--file_format', '-F', type=str, choices=['ply', 'obj', 'both'], default='ply')
    parser.add_argument('--subdivide', dest='subdivide', action='store_true')
    parser.add_argument('--shirt_length', '-S', type=float, default=SHIRT_LENGTH)
    parser.add_argument('--pant_length', '-P', type=float, default=PANT_LENGTH)
    parser.add_argument('--sleeve_length', '-L', type=float, default=SLEEVE_LENGTH)
    parser.add_argument('--mesh_set', type=str, default="set2")
    args = parser.parse_args()

    smpl_model = SMPL(model_path=f'/home/kristijan/data/hierprob3d/smpl/SMPL_{args.gender.upper()}.pkl')

    for mesh_idx, (pose_fun, shape_fun) in enumerate(SETS[args.mesh_set]):
        select_original(
            args, 
            smpl_model,
            mesh_idx,
            pose_fun(),
            shape_fun()
        )
