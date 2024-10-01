import os
import json
import torch
import numpy as np
from smplx import SMPL

from src.garment import Garment
from src.const import (
    DISPLACEMENTS,
    INIT_LEFT_BACK_PANT,
    INIT_LEFT_FRONT_PANT,
    INIT_RIGHT_BACK_PANT,
    INIT_RIGHT_FRONT_PANT,
    INIT_FRONT_RIGHT_SLEEVE,
    INIT_BACK_RIGHT_SLEEVE,
    INIT_FRONT_LEFT_SLEEVE,
    INIT_BACK_LEFT_SLEEVE,
    INIT_UPPER_BACK,
    INIT_UPPER_FRONT,
    SEAM_IDX_DICT
)
import src.const as const
from src.geometry import (
    modify_mesh_with_plane_cut
)
from src.seams import (
    determine_pant_seams,
    determine_shirt_seams,
    determine_sleeve_seams
)
from src.utils import (
    export,
    color_code_stretches,
    update_color_indices,
    extract_local_stretches
)


def export_mesh(args, verts, faces, path, file_format, color_dict):
    export(args, verts, faces, path, file_format, color_dict)


def export_stretch_arrays(design_dict, verts, faces, part_name, mesh_set_dir, latest_set_dir):
    stretch_array_u, stretch_array_v = extract_local_stretches(
        verts=verts,
        faces=faces,
        design_dict=design_dict['stretches'],
        garment_part=part_name.split('_')[0]
    )

    np.savetxt(f'{mesh_set_dir}/{part_name}/stretches_u.txt', stretch_array_u)
    np.savetxt(f'{mesh_set_dir}/{part_name}/stretches_v.txt', stretch_array_v)
    np.savetxt(f'{latest_set_dir}/{part_name}/stretches_u.txt', stretch_array_u)
    np.savetxt(f'{latest_set_dir}/{part_name}/stretches_v.txt', stretch_array_v)


def initialize_smpl_models(smpl_dir):
    smpl_models = {}
    smpl_models['male'] = SMPL(model_path=os.path.join(smpl_dir, 'SMPL_MALE.pkl'), gender='male')
    smpl_models['female'] = SMPL(model_path=os.path.join(smpl_dir, 'SMPL_FEMALE.pkl'), gender='female')
    return smpl_models


def initialize_garment_and_configs(args, smpl_models):
    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)

    verts = smpl_models['female']().vertices[0].cpu().detach().numpy()
    faces = smpl_models['female'].faces
    garment = Garment(verts, faces, skirtification_type='default', use_darts=args.use_darts)

    return garment, design_dict, set_dict


def determine_all_seams(garment, design_dict):
    verts = garment.mesh.vertices
    seam_idx_dict = SEAM_IDX_DICT['default']
    seams_info = {}

    # Upper garment (shirt) seams
    seams_info['upper_front'], y_upper_threshold = determine_shirt_seams(
        verts=verts, 
        shirt_length=design_dict['dims']['upper'], 
        seam_idx_dict=seam_idx_dict['upper_front']
    )
    seams_info['upper_back'], _ = determine_shirt_seams(
        verts=verts, 
        shirt_length=design_dict['dims']['upper'], 
        seam_idx_dict=seam_idx_dict['upper_back']
    )

    # Sleeve seams
    sleeve_parts = ['front_right', 'back_right', 'front_left', 'back_left']
    for part in sleeve_parts:
        seams, x_sleeve_threshold = determine_sleeve_seams(
            verts=verts, 
            sleeve_length=design_dict['dims']['sleeve'], 
            seam_idx_dict=seam_idx_dict[f'sleeve_{part}']
        )
        seams_info[f'sleeve_{part}'] = {
            'seams': seams,
            'threshold': x_sleeve_threshold
        }

    # Lower garment (pant) seams
    lower_parts = ['front_right', 'front_left', 'back_right', 'back_left']
    y_lower_threshold_low = None
    y_lower_threshold_up = None
    for part in lower_parts:
        seams, y_low, y_up = determine_pant_seams(
            verts=verts, 
            pant_length=design_dict['dims']['lower'], 
            seam_idx_dict=seam_idx_dict[f'lower_{part}'], 
            side=part.split('_')[1],
            pant_offset=design_dict['dims']['lower_offset']
        )
        seams_info[f'lower_{part}'] = {
            'seams': seams,
            'threshold_low': y_low,
            'threshold_up': y_up
        }
        if y_lower_threshold_low is None or y_low < y_lower_threshold_low:
            y_lower_threshold_low = y_low
        if y_lower_threshold_up is None or y_up > y_lower_threshold_up:
            y_lower_threshold_up = y_up

    # Store global thresholds
    seams_info['y_upper_threshold'] = y_upper_threshold
    seams_info['y_lower_threshold_low'] = y_lower_threshold_low
    seams_info['y_lower_threshold_up'] = y_lower_threshold_up

    return seams_info


def modify_body_mesh(garment, seams_info):
    verts = garment.mesh.vertices
    faces = garment.mesh.faces

    # Upper cut
    modified_verts = modify_mesh_with_plane_cut(
        vertices=verts,
        faces=faces,
        cutting_point=seams_info['y_upper_threshold'],
        plane_orientation='horizontal'
    )

    # Update y-coordinates after upper cut
    y_lower_threshold_low = min(v[1] for v in modified_verts if v[1] > seams_info['y_upper_threshold'])
    y_lower_threshold_up = seams_info['y_lower_threshold_up']

    # Lower cuts
    modified_verts = modify_mesh_with_plane_cut(
        vertices=modified_verts,
        faces=faces,
        cutting_point=y_lower_threshold_low,
        plane_orientation='horizontal'
    )
    if y_lower_threshold_up is not None:
        modified_verts = modify_mesh_with_plane_cut(
            vertices=modified_verts,
            faces=faces,
            cutting_point=y_lower_threshold_up,
            plane_orientation='horizontal'
        )

    # Sleeve cuts
    for side in ['left', 'right']:
        sleeve_threshold = seams_info[f'sleeve_front_{side}']['threshold']
        modified_verts = modify_mesh_with_plane_cut(
            vertices=modified_verts,
            faces=faces,
            cutting_point=sleeve_threshold,
            plane_orientation='vertical',
            sleeve_side=side
        )

    # Update seams_info with new thresholds
    seams_info['y_lower_threshold_low'] = y_lower_threshold_low

    return modified_verts, seams_info


def flood_fill_all_parts(garment, modified_verts, seams_info):
    garment_parts = {}
    
    # Upper garment (shirt) flood fill
    garment_parts['upper_front'] = garment.flood_fill_vertices(
        vertex_positions=modified_verts, 
        boundary_vertices=seams_info['upper_front'], 
        y_threshold=seams_info['y_upper_threshold'], 
        start_vertex=INIT_UPPER_FRONT
    )
    garment_parts['upper_back'] = garment.flood_fill_vertices(
        vertex_positions=modified_verts, 
        boundary_vertices=seams_info['upper_back'], 
        y_threshold=seams_info['y_upper_threshold'], 
        start_vertex=INIT_UPPER_BACK
    )

    # Sleeve flood fill
    sleeve_init_points = {
        'sleeve_front_right': INIT_FRONT_RIGHT_SLEEVE,
        'sleeve_back_right': INIT_BACK_RIGHT_SLEEVE,
        'sleeve_front_left': INIT_FRONT_LEFT_SLEEVE,
        'sleeve_back_left': INIT_BACK_LEFT_SLEEVE
    }

    for key, init_idx in sleeve_init_points.items():
        garment_parts[key] = garment.flood_fill_sleeve_vertices(
            vertex_positions=modified_verts, 
            boundary_vertices=seams_info[key]['seams'], 
            start_vertex=init_idx, 
            x_threshold=seams_info[key]['threshold'], 
            side=key.split('_')[-1]
        )

    # Lower garment (pant) flood fill
    lower_init_points = {
        'lower_front_right': INIT_RIGHT_FRONT_PANT,
        'lower_front_left': INIT_LEFT_FRONT_PANT,
        'lower_back_right': INIT_RIGHT_BACK_PANT,
        'lower_back_left': INIT_LEFT_BACK_PANT
    }

    for key, init_idx in lower_init_points.items():
        garment_parts[key] = garment.flood_fill_vertices(
            vertex_positions=modified_verts, 
            boundary_vertices=seams_info[key]['seams'],
            y_threshold=seams_info[key]['threshold_low'], 
            start_vertex=init_idx,
            up_pant_threshold=seams_info[key]['threshold_up']
        )

    return garment_parts


def create_directories(mesh_set_dir, latest_set_dir):
    garment_part_names = [
        'upper_front', 'upper_back',
        'sleeve_front_right', 'sleeve_back_right',
        'sleeve_front_left', 'sleeve_back_left',
        'lower_front_right', 'lower_front_left',
        'lower_back_right', 'lower_back_left'
    ]

    for part_name in garment_part_names:
        os.makedirs(os.path.join(mesh_set_dir, part_name), exist_ok=True)
        os.makedirs(os.path.join(latest_set_dir, part_name), exist_ok=True)

    # Create directories for body meshes and color-coded designs
    os.makedirs(os.path.join(mesh_set_dir, 'body'), exist_ok=True)
    os.makedirs(os.path.join(latest_set_dir, 'body'), exist_ok=True)
    os.makedirs(os.path.join('data/simulated/houdini/latest'), exist_ok=True)


def process_garment_set(args, modified_models, garment, design_dict, set_dict, garment_parts, offset_type):
    for set_element_idx in range(len(set_dict['poses'])):
        pose_fun = getattr(const, set_dict['poses'][set_element_idx])
        shape_fun = getattr(const, set_dict['shapes'][set_element_idx])
        gender = set_dict['genders'][set_element_idx]
        
        posed_verts = modified_models[gender](
            body_pose=pose_fun(), 
            betas=shape_fun()
        ).vertices[0].cpu().detach().numpy()

        mesh_name = 'init' if set_element_idx == 0 else f'target-{set_element_idx:02d}'
        mesh_set_dir = os.path.join('data/embedded/', f'{args.design}-{args.body_set}/{offset_type}')
        latest_set_dir = os.path.join('data/embedded/latest/', offset_type)

        # Create necessary directories
        create_directories(mesh_set_dir, latest_set_dir)

        process_and_export_garment_parts(args, garment, design_dict, garment_parts, posed_verts, offset_type, set_element_idx, mesh_name, mesh_set_dir, latest_set_dir)
        
        export_body_mesh(args, posed_verts, modified_models[gender].faces, set_element_idx, mesh_set_dir, latest_set_dir, gender)

        if mesh_name == 'init' and offset_type == 'skintight':
            export_color_coded_designs(args, garment, design_dict, garment_parts, posed_verts, mesh_set_dir, latest_set_dir)


def process_and_export_garment_parts(args, garment, design_dict, garment_parts, posed_verts, offset_type, set_element_idx, mesh_name, mesh_set_dir, latest_set_dir):
    garment_part_names = [
        'upper_front', 'upper_back',
        'sleeve_front_right', 'sleeve_back_right',
        'sleeve_front_left', 'sleeve_back_left',
        'lower_front_right', 'lower_front_left',
        'lower_back_right', 'lower_back_left'
    ]

    for part_name in garment_part_names:
        part_verts, part_faces = garment.extract_garment_mesh(
            posed_verts, 
            garment.mesh.faces, 
            garment_parts[part_name],
            offset=DISPLACEMENTS[offset_type], 
            segment_name=part_name
        )

        color_dict = {f'{part_name}_colors': {part_name: garment_parts[part_name]}}
        updated_color_dict = update_color_indices(garment_parts[part_name], color_dict[f'{part_name}_colors'])

        export_mesh(args, part_verts, part_faces, f'{mesh_set_dir}/{part_name}/{mesh_name}', args.file_format, updated_color_dict)
        export_mesh(args, part_verts, part_faces, f'{latest_set_dir}/{part_name}/{mesh_name}', args.file_format, updated_color_dict)

        export_stretch_arrays(design_dict, part_verts, part_faces, part_name, mesh_set_dir, latest_set_dir)


def export_body_mesh(args, verts, faces, set_element_idx, mesh_set_dir, latest_set_dir, gender):
    body_colors = {'gray': list(range(len(verts)))}
    export(args, verts, faces, os.path.join(mesh_set_dir, 'body', f'body-{set_element_idx:02d}'), args.file_format, body_colors)
    export(args, verts, faces, os.path.join(latest_set_dir, 'body', f'body-{set_element_idx:02d}'), args.file_format, body_colors)
    export(args, verts, faces, os.path.join('data/simulated/houdini/latest', gender), args.file_format, body_colors)


def export_color_coded_designs(args, garment, design_dict, garment_parts, posed_verts, mesh_set_dir, latest_set_dir):
    upper_indices = (garment_parts['upper_front'] + garment_parts['upper_back'] + 
                     garment_parts['sleeve_front_right'] + garment_parts['sleeve_back_right'] + 
                     garment_parts['sleeve_front_left'] + garment_parts['sleeve_back_left'])
    lower_indices = sum([garment_parts[f'lower_{part}'] for part in ['front_right', 'front_left', 'back_right', 'back_left']], [])

    upper_garment_verts, upper_garment_faces = garment.extract_garment_mesh(posed_verts, garment.mesh.faces, upper_indices, offset=DISPLACEMENTS['skintight'])
    lower_garment_verts, lower_garment_faces = garment.extract_garment_mesh(posed_verts, garment.mesh.faces, lower_indices, offset=DISPLACEMENTS['skintight'])

    upper_garment_stretch_array_u, _ = extract_local_stretches(upper_garment_verts, upper_garment_faces, design_dict['stretches'], 'upper')
    lower_garment_stretch_array_u, _ = extract_local_stretches(lower_garment_verts, lower_garment_faces, design_dict['stretches'], 'lower')

    upper_garment_mesh = color_code_stretches(upper_garment_verts, upper_garment_faces, upper_garment_stretch_array_u)
    lower_garment_mesh = color_code_stretches(lower_garment_verts, lower_garment_faces, lower_garment_stretch_array_u)

    upper_garment_mesh.export(f'{mesh_set_dir}/upper_garment_init.ply')
    lower_garment_mesh.export(f'{mesh_set_dir}/lower_garment_init.ply')
    upper_garment_mesh.export(f'{latest_set_dir}/upper_garment_init.ply')
    lower_garment_mesh.export(f'{latest_set_dir}/lower_garment_init.ply')


def initialize_modified_smpl_models(smpl_dir, modified_verts):
    modified_models = {}
    for gender in ['male', 'female']:
        modified_models[gender] = SMPL(
            model_path=os.path.join(smpl_dir, f'SMPL_{gender.upper()}.pkl'),
            gender=gender,
            v_template=torch.from_numpy(modified_verts).float()
        )
    return modified_models


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
