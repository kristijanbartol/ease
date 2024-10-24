import argparse
import os
from shutil import rmtree
import json
import subprocess

from src.dress_processing import (
    setup_directories,
    load_set_dict,
    generate_original_meshes,
    initialize_dress_garment_and_configs,
    load_skirtified_meshes,
    determine_dress_seams,
    flood_fill_dress_parts,
    store_colored_faces,
    process_skirtified_garment_set
)
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
from src.utils import export_edge_lengths
from src.vis import visualize_pattern


def select_skirtified(args):
    original_dir, skirtified_dir = setup_directories(args)
    set_dict = load_set_dict(args)

    original_meshes, smpl_models = generate_original_meshes(args.smpl_dir, original_dir, set_dict)

    if not os.path.exists(skirtified_dir):
        print('NOTE: Skirtified meshes not yet created.')
        print('NOTE: Original SMPL meshes generated. Please create skirtified meshes and run again.')
        return

    skirtified_meshes = load_skirtified_meshes(skirtified_dir)
    garment, design_dict = initialize_dress_garment_and_configs(args, skirtified_meshes[0])
    
    seams_info = determine_dress_seams(garment, design_dict)
    garment_parts = flood_fill_dress_parts(garment, seams_info)
    
    store_colored_faces(garment.mesh.vertices, garment.mesh.faces, seams_info['upper_front'], os.path.join(skirtified_dir, 'boundaries.ply'))
    store_colored_faces(garment.mesh.vertices, garment.mesh.faces, garment_parts['upper_front'], os.path.join(skirtified_dir, 'patch.ply'))

    for offset_type in ['skintight', 'loose']:
        process_skirtified_garment_set(args, skirtified_meshes, original_meshes, garment, design_dict, set_dict, garment_parts, offset_type, smpl_models)

    garment.store_seamline_vertex_pairs(subdir=f'{args.design}-{args.body_set}')
    
    return garment


def select_default(args):
    smpl_models = initialize_smpl_models(args.smpl_dir)
    garment, design_dict, set_dict = initialize_garment_and_configs(args, smpl_models)
    
    seams_info = determine_all_seams(garment, design_dict)
    modified_verts, updated_seams_info = modify_body_mesh(garment, seams_info)
    modified_models = initialize_modified_smpl_models(args.smpl_dir, modified_verts)
    
    garment_parts = flood_fill_all_parts(garment, modified_verts, updated_seams_info)
    
    for offset_type in ['skintight', 'loose']:
        process_garment_set(args, modified_models, garment, design_dict, set_dict, garment_parts, offset_type)

    garment.store_seamline_vertex_pairs(subdir=f'{args.design}-{args.body_set}')
    
    return garment
    

def run_parameterization(project_path, is_skirtified, use_darts):
    cpp_program_path = "./garment-parameterization/build/optimize_set_with_seamlines"

    root_project_path_arg = os.path.abspath(project_path)
    optim_dress_arg = "1" if is_skirtified else "0"
    use_darts_arg = "1" if use_darts else "0"

    command = [cpp_program_path, root_project_path_arg, optim_dress_arg, use_darts_arg]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the C++ program: {e}")
        print(f"Error output: {e.stderr}")
    except FileNotFoundError:
        print(f"The C++ program was not found at {cpp_program_path}")


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

    is_skirtified = False
    garment = None

    print('#1 Generating specified embedded design...')
    if not design_dict['flags']['skirtified']:
        garment = select_default(args)
    else:
        if design_dict['flags']['type'] == 'dress':
            garment = select_skirtified(args)
            is_skirtified = True
        else:
            print('If the skirtified flag is selected, the garment type should be (dress).')

    print('#2 Running parameterization...')
    run_parameterization(args.project_dir, is_skirtified, args.use_darts)

    print('#3 Visualize the optimized pattern...')
    visualize_pattern(is_skirtified)
    
    print("#4 Export the resulting optimized edge lengths")
    export_edge_lengths(garment)
    