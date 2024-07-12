'''
Use for selecting the garment submeshes.

Relevant at the point when PBS will be applied, until then, the
3D grid creation operations can be done on the original mesh.
'''

import numpy as np
import torch
import os
import json
from smplx import SMPL

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
from src.garment import Garment
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
    read_ref_shape,
    extract_local_stretches
)


def select_original(args, smpl_dir):
    male_smpl_model = SMPL(model_path=os.path.join(smpl_dir, 'SMPL_MALE.pkl'), gender='female')
    female_smpl_model = SMPL(model_path=os.path.join(smpl_dir, 'SMPL_FEMALE.pkl'), gender='female')

    # NOTE: Using the female model to find the segmentations
    # Male only in case is needed to store the body for PBS or skirtification
    verts = female_smpl_model().vertices[0].cpu().detach().numpy()
    faces = female_smpl_model.faces

    garment = Garment(verts, faces)

    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)

    # Helper structures.
    sleeve_init_points = {
        'front_right': INIT_FRONT_RIGHT_SLEEVE,
        'back_right': INIT_BACK_RIGHT_SLEEVE,
        'front_left': INIT_FRONT_LEFT_SLEEVE,
        'back_left': INIT_BACK_LEFT_SLEEVE
    }
    sleeve_indices = {}
    sleeve_thresholds = {}

    pant_init_points = {
        'front_right': INIT_RIGHT_FRONT_PANT,
        'front_left': INIT_LEFT_FRONT_PANT,
        'back_right': INIT_RIGHT_BACK_PANT,
        'back_left': INIT_LEFT_BACK_PANT
    }
    pant_indices = {}
    pant_thresholds = {}

    # Determine seams.
    seam_idxs_front, y_shirt_threshold = determine_shirt_seams(
        verts=verts, 
        shirt_length=design_dict['dims']['upper'], 
        seam_idx_dict=SEAM_IDX_DICT['upper_front']
    )
    seam_idxs_back, _ = determine_shirt_seams(
        verts=verts, 
        shirt_length=design_dict['dims']['upper'], 
        seam_idx_dict=SEAM_IDX_DICT['upper_back']
    )

    for key, init_idx in sleeve_init_points.items():
        seams, x_sleeve_threshold = determine_sleeve_seams(
            verts=verts, 
            sleeve_length=design_dict['dims']['sleeve'], 
            seam_idx_dict=SEAM_IDX_DICT[f'sleeve_{key}']
        )
        sleeve_indices[key] = seams
        sleeve_thresholds[key] = x_sleeve_threshold

    for key, init_idx in pant_init_points.items():
        seams, y_pant_threshold = determine_pant_seams(
            verts=verts, 
            pant_length=design_dict['dims']['lower'], 
            seam_idx_dict=SEAM_IDX_DICT[f'lower_{key}'], 
            side=key.split('_')[1]
        )
        pant_indices[key] = seams
        pant_thresholds[key] = y_pant_threshold

    # Modify body mesh by cutting planes.
    modified_verts = modify_mesh_with_plane_cut(
        vertices=verts,
        faces=faces,
        cutting_point=y_shirt_threshold,
        plane_orientation='horizontal'
    )
    modified_verts = modify_mesh_with_plane_cut(
        vertices=modified_verts,
        faces=faces,
        cutting_point=max(sleeve_thresholds['front_left'], sleeve_thresholds['back_left']),
        plane_orientation='vertical',
        sleeve_side='left'
    )
    modified_verts = modify_mesh_with_plane_cut(
        vertices=modified_verts,
        faces=faces,
        cutting_point=min(sleeve_thresholds['front_right'], sleeve_thresholds['back_right']),
        plane_orientation='vertical',
        sleeve_side='right'
    )

    # Flood fill.
    front_v_idxs = garment.flood_fill_vertices(
        vertex_positions=modified_verts, 
        boundary_vertices=seam_idxs_front, 
        y_threshold=y_shirt_threshold, 
        start_vertex=INIT_UPPER_FRONT
    )
    back_v_idxs = garment.flood_fill_vertices(
        vertex_positions=modified_verts, 
        boundary_vertices=seam_idxs_back, 
        y_threshold=y_shirt_threshold, 
        start_vertex=INIT_UPPER_BACK
    )
    for key, init_idx in sleeve_init_points.items():
        sleeve_indices[key] = garment.flood_fill_sleeve_vertices(
            vertex_positions=modified_verts, 
            boundary_vertices=sleeve_indices[key], 
            start_vertex=init_idx, 
            x_threshold=sleeve_thresholds[key], 
            side=key.split('_')[1]
        )
    for key, init_idx in pant_init_points.items():
        pant_indices[key] = garment.flood_fill_vertices(
            vertex_positions=modified_verts, 
            boundary_vertices=pant_indices[key],
            y_threshold=pant_thresholds[key], 
            start_vertex=init_idx
        )

    modified_male_smpl_model = SMPL(os.path.join(smpl_dir, 'SMPL_MALE.pkl'), gender='male', v_template=torch.from_numpy(modified_verts))
    modified_female_smpl_model = SMPL(os.path.join(smpl_dir, 'SMPL_FEMALE.pkl'), gender='female', v_template=torch.from_numpy(modified_verts))

    for offset_type in ['skintight', 'loose']:
        for set_element_idx in range(len(set_dict['poses'])):
            pose_fun = getattr(const, set_dict['poses'][set_element_idx])
            shape_fun = getattr(const, set_dict['shapes'][set_element_idx])
            gender = set_dict['genders'][set_element_idx]
            modified_smpl_model = modified_male_smpl_model if gender == 'male' else modified_female_smpl_model
            posed_verts = modified_smpl_model(
                body_pose=pose_fun(), 
                betas=shape_fun()
            ).vertices[0].cpu().detach().numpy()

            upper_indices = front_v_idxs + back_v_idxs + sleeve_indices['front_right'] + sleeve_indices['back_right'] + sleeve_indices['front_left'] + sleeve_indices['back_left']
            lower_indices = sum(pant_indices.values(), [])

            upper_garment_verts, upper_garment_faces = garment.extract_garment_mesh(posed_verts, faces, upper_indices, offset=DISPLACEMENTS[offset_type])
            lower_garment_verts, lower_garment_faces = garment.extract_garment_mesh(posed_verts, faces, lower_indices, offset=DISPLACEMENTS[offset_type])

            upper_front_verts, upper_front_faces = garment.extract_garment_mesh(posed_verts, faces, front_v_idxs, offset=DISPLACEMENTS[offset_type], segment_name='upper_front')
            upper_back_verts, upper_back_faces = garment.extract_garment_mesh(posed_verts, faces, back_v_idxs, offset=DISPLACEMENTS[offset_type], segment_name='upper_back')
            sleeve_front_right_verts, sleeve_front_right_faces = garment.extract_garment_mesh(posed_verts, faces, sleeve_indices['front_right'], offset=DISPLACEMENTS[offset_type], segment_name='sleeve_front_right')
            sleeve_back_right_verts, sleeve_back_right_faces = garment.extract_garment_mesh(posed_verts, faces, sleeve_indices['back_right'], offset=DISPLACEMENTS[offset_type], segment_name='sleeve_back_right')
            sleeve_front_left_verts, sleeve_front_left_faces = garment.extract_garment_mesh(posed_verts, faces, sleeve_indices['front_left'], offset=DISPLACEMENTS[offset_type], segment_name='sleeve_front_left')
            sleeve_back_left_verts, sleeve_back_left_faces = garment.extract_garment_mesh(posed_verts, faces, sleeve_indices['back_left'], offset=DISPLACEMENTS[offset_type], segment_name='sleeve_back_left')
            lower_front_right_verts, lower_front_right_faces = garment.extract_garment_mesh(posed_verts, faces, pant_indices['front_right'], offset=DISPLACEMENTS[offset_type], segment_name='lower_front_right')
            lower_front_left_verts, lower_front_left_faces = garment.extract_garment_mesh(posed_verts, faces, pant_indices['front_left'], offset=DISPLACEMENTS[offset_type], segment_name='lower_front_left')
            lower_back_right_verts, lower_back_right_faces = garment.extract_garment_mesh(posed_verts, faces, pant_indices['back_right'], offset=DISPLACEMENTS[offset_type], segment_name='lower_back_right')
            lower_back_left_verts, lower_back_left_faces = garment.extract_garment_mesh(posed_verts, faces, pant_indices['back_left'], offset=DISPLACEMENTS[offset_type], segment_name='lower_back_left')

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
                'gray': list(range(len(posed_verts)))
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
            mesh_name = 'init' if set_element_idx == 0 else f'target-{set_element_idx:02d}'
            mesh_set_dir = os.path.join('data/embedded/', f'{args.design}-{args.body_set}/{offset_type}')
            latest_set_dir = os.path.join('data/embedded/latest/', offset_type)

            os.makedirs(os.path.join(mesh_set_dir, 'upper_front/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'upper_back/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'sleeve_front_right/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'sleeve_back_right/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'sleeve_front_left/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'sleeve_back_left/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'lower_front_right/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'lower_front_left/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'lower_back_right/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'lower_back_left/'), exist_ok=True)

            os.makedirs(os.path.join(latest_set_dir, 'upper_front/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'upper_back/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'sleeve_front_right/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'sleeve_back_right/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'sleeve_front_left/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'sleeve_back_left/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'lower_front_right/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'lower_front_left/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'lower_back_right/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'lower_back_left/'), exist_ok=True)

            export(args, upper_front_verts, upper_front_faces, f'{mesh_set_dir}/upper_front/{mesh_name}', args.file_format, updated_front_shirt_colors)
            export(args, upper_back_verts, upper_back_faces, f'{mesh_set_dir}/upper_back/{mesh_name}', args.file_format, updated_back_shirt_colors)
            export(args, sleeve_front_right_verts, sleeve_front_right_faces, f'{mesh_set_dir}/sleeve_front_right/{mesh_name}', args.file_format, updated_front_right_sleeve_colors)
            export(args, sleeve_back_right_verts, sleeve_back_right_faces, f'{mesh_set_dir}/sleeve_back_right/{mesh_name}', args.file_format, updated_back_right_sleeve_colors)
            export(args, sleeve_front_left_verts, sleeve_front_left_faces, f'{mesh_set_dir}/sleeve_front_left/{mesh_name}', args.file_format, updated_front_left_sleeve_colors)
            export(args, sleeve_back_left_verts, sleeve_back_left_faces, f'{mesh_set_dir}/sleeve_back_left/{mesh_name}', args.file_format, updated_back_left_sleeve_colors)
            export(args, lower_front_right_verts, lower_front_right_faces, f'{mesh_set_dir}/lower_front_right/{mesh_name}', args.file_format, updated_front_right_pant_colors)
            export(args, lower_front_left_verts, lower_front_left_faces, f'{mesh_set_dir}/lower_front_left/{mesh_name}', args.file_format, updated_front_left_pant_colors)
            export(args, lower_back_right_verts, lower_back_right_faces, f'{mesh_set_dir}/lower_back_right/{mesh_name}', args.file_format, updated_back_right_pant_colors)
            export(args, lower_back_left_verts, lower_back_left_faces, f'{mesh_set_dir}/lower_back_left/{mesh_name}', args.file_format, updated_back_left_pant_colors)

            export(args, upper_front_verts, upper_front_faces, f'{latest_set_dir}/upper_front/{mesh_name}', args.file_format, updated_front_shirt_colors)
            export(args, upper_back_verts, upper_back_faces, f'{latest_set_dir}/upper_back/{mesh_name}', args.file_format, updated_back_shirt_colors)
            export(args, sleeve_front_right_verts, sleeve_front_right_faces, f'{latest_set_dir}/sleeve_front_right/{mesh_name}', args.file_format, updated_front_right_sleeve_colors)
            export(args, sleeve_back_right_verts, sleeve_back_right_faces, f'{latest_set_dir}/sleeve_back_right/{mesh_name}', args.file_format, updated_back_right_sleeve_colors)
            export(args, sleeve_front_left_verts, sleeve_front_left_faces, f'{latest_set_dir}/sleeve_front_left/{mesh_name}', args.file_format, updated_front_left_sleeve_colors)
            export(args, sleeve_back_left_verts, sleeve_back_left_faces, f'{latest_set_dir}/sleeve_back_left/{mesh_name}', args.file_format, updated_back_left_sleeve_colors)
            export(args, lower_front_right_verts, lower_front_right_faces, f'{latest_set_dir}/lower_front_right/{mesh_name}', args.file_format, updated_front_right_pant_colors)
            export(args, lower_front_left_verts, lower_front_left_faces, f'{latest_set_dir}/lower_front_left/{mesh_name}', args.file_format, updated_front_left_pant_colors)
            export(args, lower_back_right_verts, lower_back_right_faces, f'{latest_set_dir}/lower_back_right/{mesh_name}', args.file_format, updated_back_right_pant_colors)
            export(args, lower_back_left_verts, lower_back_left_faces, f'{latest_set_dir}/lower_back_left/{mesh_name}', args.file_format, updated_back_left_pant_colors)

            # Prepare local stretch arrays
            upper_front_stretch_array_u, upper_front_stretch_array_v = extract_local_stretches(
                verts=upper_front_verts,
                faces=upper_front_faces,
                design_dict=design_dict['stretches'],
                garment_part='upper'
            )
            upper_back_stretch_array_u, upper_back_strech_array_v = extract_local_stretches(
                verts=upper_back_verts,
                faces=upper_back_faces,
                design_dict=design_dict['stretches'],
                garment_part='upper'
            )
            sleeve_front_right_stretch_array_u, sleeve_front_right_stretch_array_v = extract_local_stretches(
                verts=sleeve_front_right_verts,
                faces=sleeve_front_right_faces,
                design_dict=design_dict['stretches'],
                garment_part='sleeve',
                side='right'
            )
            sleeve_back_right_stretch_array_u, sleeve_back_right_stretch_array_v = extract_local_stretches(
                verts=sleeve_back_right_verts,
                faces=sleeve_back_right_faces,
                design_dict=design_dict['stretches'],
                garment_part='sleeve',
                side='right'
            )
            sleeve_front_left_stretch_array_u, sleeve_front_left_stretch_array_v = extract_local_stretches(
                verts=sleeve_front_left_verts,
                faces=sleeve_front_left_faces,
                design_dict=design_dict['stretches'],
                garment_part='sleeve',
                side='left'
            )
            sleeve_back_left_stretch_array_u, sleeve_back_left_stretch_array_v = extract_local_stretches(
                verts=sleeve_back_left_verts,
                faces=sleeve_back_left_faces,
                design_dict=design_dict['stretches'],
                garment_part='sleeve',
                side='left'
            )

            lower_front_right_stretch_array_u, lower_front_right_stretch_array_v = extract_local_stretches(
                verts=lower_front_right_verts,
                faces=lower_front_right_faces,
                design_dict=design_dict['stretches'],
                garment_part='lower'
            )
            lower_back_right_stretch_array_u, lower_back_right_stretch_array_v = extract_local_stretches(
                verts=lower_back_right_verts,
                faces=lower_back_right_faces,
                design_dict=design_dict['stretches'],
                garment_part='lower'
            )
            lower_front_left_stretch_array_u, lower_front_left_stretch_array_v = extract_local_stretches(
                verts=lower_front_left_verts,
                faces=lower_front_left_faces,
                design_dict=design_dict['stretches'],
                garment_part='lower'
            )
            lower_back_left_stretch_array_u, lower_back_left_stretch_array_v = extract_local_stretches(
                verts=lower_back_left_verts,
                faces=lower_back_left_faces,
                design_dict=design_dict['stretches'],
                garment_part='lower'
            )

            np.savetxt(f'{mesh_set_dir}/upper_front/stretches_u.txt', upper_front_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/upper_back/stretches_u.txt', upper_back_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/sleeve_front_right/stretches_u.txt', sleeve_front_right_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/sleeve_back_right/stretches_u.txt', sleeve_back_right_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/sleeve_front_left/stretches_u.txt', sleeve_front_left_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/sleeve_back_left/stretches_u.txt', sleeve_back_left_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/lower_front_right/stretches_u.txt', lower_front_right_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/lower_back_right/stretches_u.txt', lower_back_right_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/lower_front_left/stretches_u.txt', lower_front_left_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/lower_back_left/stretches_u.txt', lower_back_left_stretch_array_u)

            np.savetxt(f'{mesh_set_dir}/upper_front/stretches_v.txt', upper_front_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/upper_back/stretches_v.txt', upper_back_strech_array_v)
            np.savetxt(f'{mesh_set_dir}/sleeve_front_right/stretches_v.txt', sleeve_front_right_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/sleeve_back_right/stretches_v.txt', sleeve_back_right_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/sleeve_front_left/stretches_v.txt', sleeve_front_left_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/sleeve_back_left/stretches_v.txt', sleeve_back_left_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/lower_front_right/stretches_v.txt', lower_front_right_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/lower_back_right/stretches_v.txt', lower_back_right_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/lower_front_left/stretches_v.txt', lower_front_left_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/lower_back_left/stretches_v.txt', lower_back_left_stretch_array_v)

            np.savetxt(f'{latest_set_dir}/upper_front/stretches_u.txt', upper_front_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/upper_back/stretches_u.txt', upper_back_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/sleeve_front_right/stretches_u.txt', sleeve_front_right_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/sleeve_back_right/stretches_u.txt', sleeve_back_right_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/sleeve_front_left/stretches_u.txt', sleeve_front_left_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/sleeve_back_left/stretches_u.txt', sleeve_back_left_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/lower_front_right/stretches_u.txt', lower_front_right_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/lower_back_right/stretches_u.txt', lower_back_right_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/lower_front_left/stretches_u.txt', lower_front_left_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/lower_back_left/stretches_u.txt', lower_back_left_stretch_array_u)

            np.savetxt(f'{latest_set_dir}/upper_front/stretches_v.txt', upper_front_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/upper_back/stretches_v.txt', upper_back_strech_array_v)
            np.savetxt(f'{latest_set_dir}/sleeve_front_right/stretches_v.txt', sleeve_front_right_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/sleeve_back_right/stretches_v.txt', sleeve_back_right_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/sleeve_front_left/stretches_v.txt', sleeve_front_left_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/sleeve_back_left/stretches_v.txt', sleeve_back_left_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/lower_front_right/stretches_v.txt', lower_front_right_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/lower_back_right/stretches_v.txt', lower_back_right_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/lower_front_left/stretches_v.txt', lower_front_left_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/lower_back_left/stretches_v.txt', lower_back_left_stretch_array_v)

            # Store for the initial color-coded designs.
            if mesh_name == 'init' and offset_type == 'skintight':
                upper_garment_stretch_array_u, upper_garment_stretch_array_v = extract_local_stretches(
                    verts=upper_garment_verts,
                    faces=upper_garment_faces,
                    design_dict=design_dict['stretches'],
                    garment_part='upper'
                )
                lower_garment_stretch_array_u, lower_garment_stretch_array_v = extract_local_stretches(
                    verts=lower_garment_verts,
                    faces=lower_garment_faces,
                    design_dict=design_dict['stretches'],
                    garment_part='lower'
                )
                
                upper_garment_mesh = color_code_stretches(
                    verts=upper_garment_verts, 
                    faces=upper_garment_faces,
                    stretch_array=upper_garment_stretch_array_u
                )
                lower_garment_mesh = color_code_stretches(
                    verts=lower_garment_verts, 
                    faces=lower_garment_faces,
                    stretch_array=lower_garment_stretch_array_u
                )
                upper_garment_mesh.export(f'{mesh_set_dir}/upper_garment_{mesh_name}.ply')
                lower_garment_mesh.export(f'{mesh_set_dir}/lower_garment_{mesh_name}.ply')
                upper_garment_mesh.export(f'{latest_set_dir}/upper_garment_{mesh_name}.ply')
                lower_garment_mesh.export(f'{latest_set_dir}/lower_garment_{mesh_name}.ply')

            original_smpl_model = male_smpl_model if gender == 'male' else female_smpl_model
            original_verts = original_smpl_model(
                body_pose=pose_fun(), 
                betas=shape_fun()
            ).vertices[0].cpu().detach().numpy()

            export(args, original_verts, faces, f'{mesh_set_dir}/body-{set_element_idx:02d}', args.file_format, body_colors)
            export(args, original_verts, faces, f'{latest_set_dir}/body-{set_element_idx:02d}', args.file_format, body_colors)

            garment.store_seamline_vertex_pairs(subdir=f'{args.design}-{args.body_set}')
