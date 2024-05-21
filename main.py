from smplx import SMPL
import argparse

from const import (
    PANT_LENGTH,
    SHIRT_LENGTH,
    SLEEVE_LENGTH
)
from selector import select_original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gender', '-G', type=str, choices=['male', 'female', 'neutral'], default='female')
    parser.add_argument('--file_format', '-F', type=str, choices=['ply', 'obj', 'both'], default='ply')
    parser.add_argument('--subdivide', dest='subdivide', action='store_true')
    parser.add_argument('--shirt_length', '-S', type=float, default=SHIRT_LENGTH)
    parser.add_argument('--pant_length', '-P', type=float, default=PANT_LENGTH)
    parser.add_argument('--sleeve_length', '-L', type=float, default=SLEEVE_LENGTH)
    parser.add_argument('--mesh_set', type=str, default="set2")
    parser.add_argument('--os', type=str, default="linux")
    args = parser.parse_args()

    if args.os == 'macos':
        smpl_model = SMPL(model_path=f'/Users/kristijanbartol/Documents/data/hood_data/aux_data/smpl/SMPL_{args.gender.upper()}.pkl')
    else:
        smpl_model = SMPL(model_path=f'/home/kristijan/data/hierprob3d/smpl/SMPL_{args.gender.upper()}.pkl')

    select_original(
        args, 
        smpl_model
    )
