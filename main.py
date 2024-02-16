from smplx import SMPL
import argparse
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
    parser.add_argument('--file_format', '-F', type=str, choices=['ply', 'obj'], default='ply')
    parser.add_argument('--subdivide', dest='subdivide', action='store_true')
    parser.add_argument('--shirt_length', '-S', type=float, default=SHIRT_LENGTH)
    parser.add_argument('--pant_length', '-P', type=float, default=PANT_LENGTH)
    parser.add_argument('--sleeve_length', '-L', type=float, default=SLEEVE_LENGTH)
    args = parser.parse_args()

    smpl_model = SMPL(model_path='/data/hierprob3d/smpl/SMPL_FEMALE.pkl')

    if args.subdivide:
        select_subdivided(args, smpl_model)
    else:
        select_original(args, smpl_model)
