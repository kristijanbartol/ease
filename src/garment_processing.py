import json
import os

from src import const
from src.const import (
    DISPLACEMENTS,
    INIT_UPPER_FRONT,
    INIT_UPPER_BACK,
    INIT_FRONT_RIGHT_SLEEVE,
    INIT_FRONT_LEFT_SLEEVE,
    INIT_BACK_LEFT_SLEEVE,
    INIT_BACK_RIGHT_SLEEVE,
    INIT_LEFT_BACK_PANT,
    INIT_RIGHT_BACK_PANT,
    INIT_LEFT_FRONT_PANT,
    INIT_RIGHT_FRONT_PANT
)
from src.garment import Garment
from src.utils import (
    create_directories,
    export,
    export_body_mesh,
    export_color_coded_designs,
    export_stretch_arrays,
    update_color_indices
)


def initialize_garment_and_configs(args, smpl_models):
    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)

    verts = smpl_models['female']().vertices[0].cpu().detach().numpy()
    faces = smpl_models['female'].faces
    garment = Garment(verts, faces, skirtification_type='default', use_darts=args.use_darts)

    return garment, design_dict, set_dict


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

        export(args, part_verts, part_faces, f'{mesh_set_dir}/{part_name}/{mesh_name}', args.file_format, updated_color_dict)
        export(args, part_verts, part_faces, f'{latest_set_dir}/{part_name}/{mesh_name}', args.file_format, updated_color_dict)

        export_stretch_arrays(design_dict, part_verts, part_faces, part_name, mesh_set_dir, latest_set_dir)
