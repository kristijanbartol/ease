import argparse
import os
from shutil import rmtree
import json

#from src.selector_dress import select_skirtified_dress
from src.selector_dress_sonnet import select_skirtified_dress
from src.body_processing import (
    initialize_smpl_models,
    initialize_modified_smpl_models,
    modify_body_mesh
)
from src.garment_processing import (
    flood_fill_all_parts,
    initialize_garment_and_configs,
    process_garment_set
)
from src.seams import (
    determine_all_seams
)


def select_original(args, smpl_dir):
    smpl_models = initialize_smpl_models(smpl_dir)
    garment, design_dict, set_dict = initialize_garment_and_configs(args, smpl_models)
    
    seams_info = determine_all_seams(garment, design_dict)
    modified_verts, updated_seams_info = modify_body_mesh(garment, seams_info)
    modified_models = initialize_modified_smpl_models(smpl_dir, modified_verts)
    
    garment_parts = flood_fill_all_parts(garment, modified_verts, updated_seams_info)
    
    for offset_type in ['skintight', 'loose']:
        process_garment_set(args, modified_models, garment, design_dict, set_dict, garment_parts, offset_type)

    garment.store_seamline_vertex_pairs(subdir=f'{args.design}-{args.body_set}')


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
