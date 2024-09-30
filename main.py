import argparse
import os
from shutil import rmtree
import json

#from src.selector import select_original
from src.selector_sonnet import select_original
from src.selector_dress import select_skirtified_dress


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate', '-R', action='store_true', dest='regenerate')
    parser.add_argument('--use_darts', action='store_true', dest='use_darts')
    parser.add_argument('--file_format', '-F', type=str, choices=['ply', 'obj', 'both'], default='ply')
    parser.add_argument('--design', '-D', type=str, default='default')
    parser.add_argument('--body_set', type=str, default="set2")
    parser.add_argument('--os', type=str, default="linux")
    parser.add_argument('--standard_export', action='store_true', dest='standard_export')
    args = parser.parse_args()

    # TODO: Anonymize the paths before submission!!!
    if args.os == 'macos':
        smpl_dir = '/Users/kristijanbartol/Documents/data/hood_data/aux_data/smpl/'
    else:
        smpl_dir = '/home/kristijan/data/smpl/models/'

    if os.path.exists('data/embedded/latest/'):
        rmtree('data/embedded/latest/')
    if os.path.exists('data/seamlines/latest/'):
        rmtree('data/seamlines/latest/')

    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)

    if not design_dict['flags']['skirtified']:
        select_original(args, smpl_dir)
    else:
        if design_dict['flags']['type'] == 'dress':
            select_skirtified_dress(args, smpl_dir)
        else:
            pass
