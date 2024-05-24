from smplx import SMPL
import argparse

from selector import select_original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gender', '-G', type=str, choices=['male', 'female', 'neutral'], default='female')
    parser.add_argument('--file_format', '-F', type=str, choices=['ply', 'obj', 'both'], default='ply')
    parser.add_argument('--design', '-D', type=str, default='default')
    parser.add_argument('--mesh_set', type=str, default="set2")
    parser.add_argument('--os', type=str, default="linux")
    args = parser.parse_args()

    # TODO: Anonymize the paths before submission!!!
    if args.os == 'macos':
        smpl_path = f'/Users/kristijanbartol/Documents/data/hood_data/aux_data/smpl/SMPL_{args.gender.upper()}.pkl'
    else:
        smpl_path = f'/home/kristijan/data/hierprob3d/smpl/SMPL_{args.gender.upper()}.pkl'

    smpl_model = SMPL(smpl_path)

    select_original(
        args, 
        smpl_model,
        smpl_path
    )
