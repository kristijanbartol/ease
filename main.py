import argparse

from src.selector import select_original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate', '-R', action='store_true', dest='regenerate')
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
        smpl_dir = '/home/kristijan/data/hierprob3d/smpl/'

    select_original(
        args,
        smpl_dir
    )
