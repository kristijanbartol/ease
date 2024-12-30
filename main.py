import argparse
import os
from shutil import rmtree
import json

from tailorlang.mesh_processing import (
    initialize_smpl_models,
    initialize_modified_smpl_models,
    modify_body_mesh,
    apply_all_cuts,
    apply_masks,
    apply_length_params,
    apply_pre_seams,
    collect_vert_idxs_of_full_garments,
    flood_fill_all_parts,
    init_garment_for_preselect,
    init_static,
    initialize_garment_and_configs,
    process_garment_set,
    preselect
)
from tailorlang.pybind import apply_remesh
from tailorlang.seams import (
    determine_all_seams
)
from tailorlang.submodules import run_parameterization
from tailorlang.io import (
    export,
    export_edge_lengths,
    export_stretch_arrays,
    load_preselected,
    store_preselected
)
from eval.qualitative import visualize_pattern


def prepare(args):
    #design_dict, set_dict, smpl_model, canonical_verts = init_static(args)
    patch_idxs_dict, modified_verts, smpl_model = preselect(args, set_dict)
    garment_params = get_garment_params(args)
    
    modified_verts, threshold_dict = apply_length_params(garment_params, smpl_model)
    patch_idxs_dict = apply_masks(patch_idxs_dict, modified_verts, threshold_dict)
    
    if args.apply_remesh:
        patch_dict, seams_dict = apply_remesh(patch_dict, seams_dict)
    # TODO: Construct seams dict even if the remesh is not applied.
    # NOTE: To simplify, use only (begin, end) keypoints for each border and extract individual seamline vertices.
        
    stretch_arrays = construct_stretch_arrays(patch_dict, design_dict)
    seam_pairs_dict = construct_seam_pairs(seams_dict)
    
    return patch_dict, seam_pairs_dict, stretch_arrays


def select(args):
    smpl_models = initialize_smpl_models(args.smpl_dir)
    garment, design_dict, set_dict = initialize_garment_and_configs(args, smpl_models)
    
    seams_info = determine_all_seams(garment, design_dict)
    modified_verts, updated_seams_info = modify_body_mesh(garment, seams_info)
    modified_models = initialize_modified_smpl_models(args.smpl_dir, modified_verts)
    
    garment_parts = flood_fill_all_parts(garment, modified_verts, updated_seams_info)
    
    process_garment_set(args, modified_models, garment, design_dict, set_dict, garment_parts)
    
    export_stretch_arrays(design_dict, part_verts, part_faces, part_name, mesh_set_dir, latest_set_dir)

    garment.store_seamline_vertex_pairs(subdir=f'{args.design}-{args.body_set}')
    
    return garment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_darts', action='store_true', dest='use_darts',
                        help='whether to use darts in the design and parameterization algorithm')
    parser.add_argument('--file_format', '-F', type=str, choices=['ply', 'obj', 'both'], default='ply',
                        help='')
    parser.add_argument('--design', '-D', type=str, default='default')
    parser.add_argument('--body_set', type=str, default="set2")
    parser.add_argument('--project_dir', type=str, default='/home/kristijan/TailorLang/', 
                        help='an absolute path to this project')
    parser.add_argument('--smpl_dir', type=str, default="/home/kristijan/data/smpl/models/")
    parser.add_argument('--standard_export', action='store_true', dest='standard_export')
    args = parser.parse_args()

    if os.path.exists('data/embedded/latest/'):
        rmtree('data/embedded/latest/')
    if os.path.exists('data/seamlines/latest/'):
        rmtree('data/seamlines/latest/')

    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
        
    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)

    garment = None
    
    print('#0 Pre-selecting the garment meshes...')
    upper_idxs, lower_idxs, modified_verts = preselect_full_garments(set_dict)

    print('#1 Generating specified embedded design...')
    garment = select(args)

    print('#2 Running parameterization...')
    run_parameterization(args.project_dir, args.use_darts)

    print('#3 Visualize the optimized pattern...')
    visualize_pattern()
    
    print("#4 Export the resulting optimized edge lengths")
    export_edge_lengths(garment)
    