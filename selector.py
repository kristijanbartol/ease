'''
Use for selecting the garment submeshes.

Relevant at the point when PBS will be applied, until then, the
3D grid creation operations can be done on the original mesh.
'''

import trimesh
import numpy as np
import os

from const import (
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
    KEYPOINTS,
    PANT_LENGTH,
    SEAM_IDX_DICT,
    SHIRT_LENGTH,
    SLEEVE_LENGTH
)
from garment import Garment
from seams import (
    determine_pant_seams,
    determine_shirt_seams,
    determine_sleeve_seams
)
from utils import (
    export,
    color_code_stretches,
    update_color_indices
)
from mesh_sets import SETS


def select_original(args, smpl_model):
    verts = smpl_model().vertices[0].cpu().detach().numpy()
    faces = smpl_model.faces

    garment = Garment(verts, faces)

    # For the front shirt
    seam_idxs_front, y_shirt_threshold = determine_shirt_seams(verts, SHIRT_LENGTH, SEAM_IDX_DICT['upper_front'])
    front_v_idxs = garment.flood_fill_vertices(verts, seam_idxs_front, y_shirt_threshold, INIT_UPPER_FRONT)
    
    # For the back shirt
    seam_idxs_back, _ = determine_shirt_seams(verts, SHIRT_LENGTH, SEAM_IDX_DICT['upper_back'])
    back_v_idxs = garment.flood_fill_vertices(verts, seam_idxs_back, y_shirt_threshold, INIT_UPPER_BACK)

    # Process the sleeves
    sleeve_seams_and_thresholds = {
        'front_right': INIT_FRONT_RIGHT_SLEEVE,
        'back_right': INIT_BACK_RIGHT_SLEEVE,
        'front_left': INIT_FRONT_LEFT_SLEEVE,
        'back_left': INIT_BACK_LEFT_SLEEVE
    }
    sleeve_indices = {}
    for key, init_idx in sleeve_seams_and_thresholds.items():
        seams, x_sleeve_threshold = determine_sleeve_seams(verts, SLEEVE_LENGTH, SEAM_IDX_DICT[f'sleeve_{key}'])
        side = key.split('_')[1]
        sleeve_indices[key] = garment.flood_fill_sleeve_vertices(verts, seams, init_idx, x_sleeve_threshold, side)

    # Process the pants
    pant_seams_and_thresholds = {
        'front_right': (INIT_RIGHT_FRONT_PANT, 'right'),
        'front_left': (INIT_LEFT_FRONT_PANT, 'left'),
        'back_right': (INIT_RIGHT_BACK_PANT, 'right'),
        'back_left': (INIT_LEFT_BACK_PANT, 'left')
    }
    pant_indices = {}
    for key, (init_idx, side) in pant_seams_and_thresholds.items():
        seams, y_pant_threshold = determine_pant_seams(verts, PANT_LENGTH, SEAM_IDX_DICT[f'pant_{key}'], side)
        pant_indices[key] = garment.flood_fill_vertices(verts, seams, y_pant_threshold, init_idx)



    offset_distance = DISPLACEMENTS
    for mesh_idx, (pose_fun, shape_fun) in enumerate(SETS[args.mesh_set]):

        verts = smpl_model(body_pose=pose_fun(), betas=shape_fun()).vertices[0].cpu().detach().numpy()


        # Create upper and lower garment meshes
        upper_indices = front_v_idxs + back_v_idxs + sleeve_indices['front_right'] + sleeve_indices['back_right'] + sleeve_indices['front_left'] + sleeve_indices['back_left']
        lower_indices = sum(pant_indices.values(), [])

        upper_garment_verts, upper_garment_faces = garment.extract_garment_mesh(verts, faces, upper_indices, offset=offset_distance)
        lower_garment_verts, lower_garment_faces = garment.extract_garment_mesh(verts, faces, lower_indices, offset=offset_distance)

        front_shirt_verts, front_shirt_faces = garment.extract_garment_mesh(verts, faces, front_v_idxs, offset=offset_distance)
        back_shirt_verts, back_shirt_faces = garment.extract_garment_mesh(verts, faces, back_v_idxs, offset=offset_distance)
        front_right_sleeve_verts, front_right_sleeve_faces = garment.extract_garment_mesh(verts, faces, sleeve_indices['front_right'], offset=offset_distance)
        back_right_sleeve_verts, back_right_sleeve_faces = garment.extract_garment_mesh(verts, faces, sleeve_indices['back_right'], offset=offset_distance)
        front_left_sleeve_verts, front_left_sleeve_faces = garment.extract_garment_mesh(verts, faces, sleeve_indices['front_left'], offset=offset_distance)
        back_left_sleeve_verts, back_left_sleeve_faces = garment.extract_garment_mesh(verts, faces, sleeve_indices['back_left'], offset=offset_distance)
        front_right_pant_verts, front_right_pant_faces = garment.extract_garment_mesh(verts, faces, pant_indices['front_right'], offset=offset_distance)
        front_left_pant_verts, front_left_pant_faces = garment.extract_garment_mesh(verts, faces, pant_indices['front_left'], offset=offset_distance)
        back_right_pant_verts, back_right_pant_faces = garment.extract_garment_mesh(verts, faces, pant_indices['back_right'], offset=offset_distance)
        back_left_pant_verts, back_left_pant_faces = garment.extract_garment_mesh(verts, faces, pant_indices['back_left'], offset=offset_distance)

        # TODO: Update the stretches array to take non-trivial distribution.
        upper_garment_mesh = color_code_stretches(
            verts=upper_garment_verts, 
            faces=upper_garment_faces,
            stretch_array=np.ones(upper_garment_faces.shape[0]) * 1.1
        )
        lower_garment_mesh = color_code_stretches(
            verts=lower_garment_verts, 
            faces=lower_garment_faces,
            stretch_array=np.ones(lower_garment_faces.shape[0]) * 1.1
        )

        front_shirt_colors = {
            'red': front_v_idxs,
        }
        back_shirt_colors = {
            'blue': back_v_idxs,
        }
        front_right_sleeve_colors = {
            'light_green': sleeve_indices['front_right'],
        }
        back_right_sleeve_colors = {
            'dark_green': sleeve_indices['back_right'],
        }
        front_left_sleeve_colors = {
            'brown': sleeve_indices['front_left']
        }
        back_left_sleeve_colors = {
            'white': sleeve_indices['back_left']
        }

        front_right_pant_colors = {
            'dark_blue': pant_indices['front_right'],
        }
        front_left_pant_colors = {
            'light_blue': pant_indices['front_left'],
        }
        back_right_pant_colors = {
            'orange': pant_indices['back_right'],
        }
        back_left_pant_colors = {
            'yellow': pant_indices['back_left']
        }
        body_colors = {
            'gray': list(range(len(verts)))
        }

        updated_front_shirt_colors = update_color_indices(front_v_idxs, front_shirt_colors)
        updated_back_shirt_colors = update_color_indices(back_v_idxs, back_shirt_colors)
        updated_front_right_sleeve_colors = update_color_indices(sleeve_indices['front_right'], front_right_sleeve_colors)
        updated_back_right_sleeve_colors = update_color_indices(sleeve_indices['back_right'], back_right_sleeve_colors)
        updated_front_left_sleeve_colors = update_color_indices(sleeve_indices['front_left'], front_left_sleeve_colors)
        updated_back_left_sleeve_colors = update_color_indices(sleeve_indices['back_left'], back_left_sleeve_colors)

        updated_front_right_pant_colors = update_color_indices(pant_indices['front_right'], front_right_pant_colors)
        updated_front_left_pant_colors = update_color_indices(pant_indices['front_left'], front_left_pant_colors)
        updated_back_right_pant_colors = update_color_indices(pant_indices['back_right'], back_right_pant_colors)
        updated_back_left_pant_colors = update_color_indices(pant_indices['back_left'], back_left_pant_colors)

        # Export garment component meshes
        mesh_name = 'init' if mesh_idx == 0 else f'target-{mesh_idx:02d}'
        mesh_set_dir = os.path.join(f'results/target_meshes/{args.mesh_set}/')
        if not os.path.exists(mesh_set_dir):
            os.makedirs(os.path.join(mesh_set_dir, 'front_shirt/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'back_shirt/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'front_right_sleeve/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'back_right_sleeve/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'front_left_sleeve/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'back_left_sleeve/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'front_right_pant/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'front_left_pant/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'back_right_pant/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'back_left_pant/'), exist_ok=True)
        upper_garment_mesh.export(f'{mesh_set_dir}/upper_garment_{mesh_name}.ply')
        lower_garment_mesh.export(f'{mesh_set_dir}/lower_garment_{mesh_name}.ply')

        export(front_shirt_verts, front_shirt_faces, updated_front_shirt_colors, f'{mesh_set_dir}/front_shirt/{mesh_name}', args.file_format)
        export(back_shirt_verts, back_shirt_faces, updated_back_shirt_colors, f'{mesh_set_dir}/back_shirt/{mesh_name}', args.file_format)
        export(front_right_sleeve_verts, front_right_sleeve_faces, updated_front_right_sleeve_colors, f'{mesh_set_dir}/front_right_sleeve/{mesh_name}', args.file_format)
        export(back_right_sleeve_verts, back_right_sleeve_faces, updated_back_right_sleeve_colors, f'{mesh_set_dir}/back_right_sleeve/{mesh_name}', args.file_format)
        export(front_left_sleeve_verts, front_left_sleeve_faces, updated_front_left_sleeve_colors, f'{mesh_set_dir}/front_left_sleeve/{mesh_name}', args.file_format)
        export(back_left_sleeve_verts, back_left_sleeve_faces, updated_back_left_sleeve_colors, f'{mesh_set_dir}/back_left_sleeve/{mesh_name}', args.file_format)

        export(front_right_pant_verts, front_right_pant_faces, updated_front_right_pant_colors, f'{mesh_set_dir}/front_right_pant/{mesh_name}', args.file_format)
        export(front_left_pant_verts, front_left_pant_faces, updated_front_left_pant_colors, f'{mesh_set_dir}/front_left_pant/{mesh_name}', args.file_format)
        export(back_right_pant_verts, back_right_pant_faces, updated_back_right_pant_colors, f'{mesh_set_dir}/back_right_pant/{mesh_name}', args.file_format)
        export(back_left_pant_verts, back_left_pant_faces, updated_back_left_pant_colors, f'{mesh_set_dir}/back_left_pant/{mesh_name}', args.file_format)

        export(verts, faces, body_colors, f'{mesh_set_dir}/body-{mesh_idx:02d}', args.file_format)
