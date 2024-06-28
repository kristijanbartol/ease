import argparse
import json

from src.selector_new import generate_garment_components


def load_config(args):
    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)
    return {
        'design': design_dict,
        'body_set': set_dict
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate', '-R', action='store_true', dest='regenerate')
    parser.add_argument('--file_format', '-F', type=str, choices=['ply', 'obj', 'both'], default='ply')
    parser.add_argument('--design', '-D', type=str, default='default')
    parser.add_argument('--body_set', type=str, default="set1")
    parser.add_argument('--segment_set', type=str, choices=['default', 'half'], default='default',
                        help='which segment set to prepare (important for loading the vertex boundary indices)')
    parser.add_argument('--os', type=str, default="linux")
    parser.add_argument('--standard_export', action='store_true', dest='standard_export')
    args = parser.parse_args()

    # TODO: Anonymize the paths before submission!!!
    if args.os == 'macos':
        smpl_dir = '/Users/kristijanbartol/Documents/data/hood_data/aux_data/smpl/'
    else:
        smpl_dir = '/home/kristijan/data/hierprob3d/smpl/'

    config = load_config(args)

    generate_garment_components(
        args,
        smpl_dir,
        config=config
    )
