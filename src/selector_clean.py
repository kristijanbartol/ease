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
    SEAM_IDX_DICT,
    INIT_IDXS,
    SEGMENT_SETS,
)
import src.const as const
from src.garment import Garment
from src.geometry import (
    modify_mesh_with_plane_cut,
    cut_seamlines
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
    set_local_stretches
)


def select_original(args, smpl_dir):
    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
    with open(f'config/sets/{args.set}.json', 'r') as json_file:
        set_dict = json.load(json_file)

    # NOTE: Using the female model to find the segmentations
    smpl_path = os.path.join(smpl_dir, 'SMPL_FEMALE.pkl')
    smpl_model = SMPL(model_path=smpl_path, gender='female')
    verts = smpl_model().vertices[0].cpu().detach().numpy()
    faces = smpl_model.faces

    garment = Garment(verts, faces)

    for segment_label in SEGMENT_SETS[args.segment_set]:
        boundary_verts = []
        for boundary_name in SEAM_IDX_DICT[segment_label]:
            boundary_verts += SEAM_IDX_DICT[segment_label][boundary_name]
        segment_idxs = garment.flood_fill_vertices_simplified(
            boundary_vertices=boundary_verts,
            start_vertex=INIT_IDXS[segment_label]
        )

    modified_verts, threshold_dict = cut_seamlines(
        design_dict=design_dict,
        verts=verts,
        faces=faces
    )

    
    '''
    # Flood fill.
    front_v_idxs = garment.flood_fill_vertices(
        vertex_positions=verts, 
        boundary_vertices=seam_idxs_front, 
        y_threshold=y_shirt_threshold, 
        start_vertex=INIT_UPPER_FRONT
    )
    back_v_idxs = garment.flood_fill_vertices(
        vertex_positions=verts, 
        boundary_vertices=seam_idxs_back, 
        y_threshold=y_shirt_threshold, 
        start_vertex=INIT_UPPER_BACK
    )
    for key, init_idx in sleeve_init_points.items():
        sleeve_indices[key] = garment.flood_fill_sleeve_vertices(
            vertex_positions=verts, 
            boundary_vertices=sleeve_indices[key], 
            start_vertex=init_idx, 
            x_threshold=sleeve_thresholds[key], 
            side=key.split('_')[1]
        )
    for key, init_idx in pant_init_points.items():
        pant_indices[key] = garment.flood_fill_vertices(
            vertex_positions=verts, 
            boundary_vertices=pant_indices[key],
            y_threshold=pant_thresholds[key], 
            start_vertex=init_idx
        )
    '''

    male_smpl_model = SMPL(os.path.join(smpl_dir, 'SMPL_MALE.pkl'), gender='male', v_template=torch.from_numpy(verts))
    female_smpl_model = SMPL(os.path.join(smpl_dir, 'SMPL_FEMALE.pkl'), gender='female', v_template=torch.from_numpy(verts))

    for offset_type in ['skintight', 'loose']:
        for set_element_idx in range(len(set_dict['poses'])):
            pose_fun = getattr(const, set_dict['poses'][set_element_idx])
            betas = read_ref_shape(set_dict['shapes'][set_element_idx]) if args.use_smplx else getattr(const, set_dict['shapes'][set_element_idx])()
            gender = set_dict['genders'][set_element_idx]
            smpl_model = male_smpl_model if gender == 'male' else female_smpl_model
            verts = smpl_model(
                body_pose=pose_fun(), 
                betas=betas
            ).vertices[0].cpu().detach().numpy()

            front_shirt_verts, front_shirt_faces = garment.extract_garment_mesh(verts, faces, front_v_idxs, offset=DISPLACEMENTS[offset_type])
            back_shirt_verts, back_shirt_faces = garment.extract_garment_mesh(verts, faces, back_v_idxs, offset=DISPLACEMENTS[offset_type])
            front_right_sleeve_verts, front_right_sleeve_faces = garment.extract_garment_mesh(verts, faces, sleeve_indices['front_right'], offset=DISPLACEMENTS[offset_type])
            back_right_sleeve_verts, back_right_sleeve_faces = garment.extract_garment_mesh(verts, faces, sleeve_indices['back_right'], offset=DISPLACEMENTS[offset_type])
            front_left_sleeve_verts, front_left_sleeve_faces = garment.extract_garment_mesh(verts, faces, sleeve_indices['front_left'], offset=DISPLACEMENTS[offset_type])
            back_left_sleeve_verts, back_left_sleeve_faces = garment.extract_garment_mesh(verts, faces, sleeve_indices['back_left'], offset=DISPLACEMENTS[offset_type])
            front_right_pant_verts, front_right_pant_faces = garment.extract_garment_mesh(verts, faces, pant_indices['front_right'], offset=DISPLACEMENTS[offset_type])
            front_left_pant_verts, front_left_pant_faces = garment.extract_garment_mesh(verts, faces, pant_indices['front_left'], offset=DISPLACEMENTS[offset_type])
            back_right_pant_verts, back_right_pant_faces = garment.extract_garment_mesh(verts, faces, pant_indices['back_right'], offset=DISPLACEMENTS[offset_type])
            back_left_pant_verts, back_left_pant_faces = garment.extract_garment_mesh(verts, faces, pant_indices['back_left'], offset=DISPLACEMENTS[offset_type])       

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
            mesh_name = 'init' if set_element_idx == 0 else f'target-{set_element_idx:02d}'
            mesh_set_dir = os.path.join(f'data/{args.design}-{args.set}/{offset_type}')
            latest_set_dir = os.path.join(f'data/latest/{offset_type}')

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

            os.makedirs(os.path.join(latest_set_dir, 'front_shirt/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'back_shirt/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'front_right_sleeve/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'back_right_sleeve/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'front_left_sleeve/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'back_left_sleeve/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'front_right_pant/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'front_left_pant/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'back_right_pant/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'back_left_pant/'), exist_ok=True)

            export(args, front_shirt_verts, front_shirt_faces, updated_front_shirt_colors, f'{mesh_set_dir}/front_shirt/{mesh_name}', args.file_format)
            export(args, back_shirt_verts, back_shirt_faces, updated_back_shirt_colors, f'{mesh_set_dir}/back_shirt/{mesh_name}', args.file_format)
            export(args, front_right_sleeve_verts, front_right_sleeve_faces, updated_front_right_sleeve_colors, f'{mesh_set_dir}/front_right_sleeve/{mesh_name}', args.file_format)
            export(args, back_right_sleeve_verts, back_right_sleeve_faces, updated_back_right_sleeve_colors, f'{mesh_set_dir}/back_right_sleeve/{mesh_name}', args.file_format)
            export(args, front_left_sleeve_verts, front_left_sleeve_faces, updated_front_left_sleeve_colors, f'{mesh_set_dir}/front_left_sleeve/{mesh_name}', args.file_format)
            export(args, back_left_sleeve_verts, back_left_sleeve_faces, updated_back_left_sleeve_colors, f'{mesh_set_dir}/back_left_sleeve/{mesh_name}', args.file_format)
            export(args, front_right_pant_verts, front_right_pant_faces, updated_front_right_pant_colors, f'{mesh_set_dir}/front_right_pant/{mesh_name}', args.file_format)
            export(args, front_left_pant_verts, front_left_pant_faces, updated_front_left_pant_colors, f'{mesh_set_dir}/front_left_pant/{mesh_name}', args.file_format)
            export(args, back_right_pant_verts, back_right_pant_faces, updated_back_right_pant_colors, f'{mesh_set_dir}/back_right_pant/{mesh_name}', args.file_format)
            export(args, back_left_pant_verts, back_left_pant_faces, updated_back_left_pant_colors, f'{mesh_set_dir}/back_left_pant/{mesh_name}', args.file_format)

            export(args, front_shirt_verts, front_shirt_faces, updated_front_shirt_colors, f'{latest_set_dir}/front_shirt/{mesh_name}', args.file_format)
            export(args, back_shirt_verts, back_shirt_faces, updated_back_shirt_colors, f'{latest_set_dir}/back_shirt/{mesh_name}', args.file_format)
            export(args, front_right_sleeve_verts, front_right_sleeve_faces, updated_front_right_sleeve_colors, f'{latest_set_dir}/front_right_sleeve/{mesh_name}', args.file_format)
            export(args, back_right_sleeve_verts, back_right_sleeve_faces, updated_back_right_sleeve_colors, f'{latest_set_dir}/back_right_sleeve/{mesh_name}', args.file_format)
            export(args, front_left_sleeve_verts, front_left_sleeve_faces, updated_front_left_sleeve_colors, f'{latest_set_dir}/front_left_sleeve/{mesh_name}', args.file_format)
            export(args, back_left_sleeve_verts, back_left_sleeve_faces, updated_back_left_sleeve_colors, f'{latest_set_dir}/back_left_sleeve/{mesh_name}', args.file_format)
            export(args, front_right_pant_verts, front_right_pant_faces, updated_front_right_pant_colors, f'{latest_set_dir}/front_right_pant/{mesh_name}', args.file_format)
            export(args, front_left_pant_verts, front_left_pant_faces, updated_front_left_pant_colors, f'{latest_set_dir}/front_left_pant/{mesh_name}', args.file_format)
            export(args, back_right_pant_verts, back_right_pant_faces, updated_back_right_pant_colors, f'{latest_set_dir}/back_right_pant/{mesh_name}', args.file_format)
            export(args, back_left_pant_verts, back_left_pant_faces, updated_back_left_pant_colors, f'{latest_set_dir}/back_left_pant/{mesh_name}', args.file_format)

            # Prepare local stretch arrays
            front_shirt_stretch_array_u, front_shirt_stretch_array_v = set_local_stretches(
                verts=front_shirt_verts,
                faces=front_shirt_faces,
                design_dict=design_dict['stretches'],
                garment_part='upper'
            )
            back_shirt_strech_array_u, back_shirt_strech_array_v = set_local_stretches(
                verts=back_shirt_verts,
                faces=back_shirt_faces,
                design_dict=design_dict['stretches'],
                garment_part='upper'
            )
            front_right_sleeve_stretch_array_u, front_right_sleeve_stretch_array_v = set_local_stretches(
                verts=front_right_sleeve_verts,
                faces=front_right_sleeve_faces,
                design_dict=design_dict['stretches'],
                garment_part='sleeves',
                side='right'
            )
            back_right_sleeve_stretch_array_u, back_right_sleeve_stretch_array_v = set_local_stretches(
                verts=back_right_sleeve_verts,
                faces=back_right_sleeve_faces,
                design_dict=design_dict['stretches'],
                garment_part='sleeves',
                side='right'
            )
            front_left_sleeve_stretch_array_u, front_left_sleeve_stretch_array_v = set_local_stretches(
                verts=front_left_sleeve_verts,
                faces=front_left_sleeve_faces,
                design_dict=design_dict['stretches'],
                garment_part='sleeves',
                side='left'
            )
            back_left_sleeve_stretch_array_u, back_left_sleeve_stretch_array_v = set_local_stretches(
                verts=back_left_sleeve_verts,
                faces=back_left_sleeve_faces,
                design_dict=design_dict['stretches'],
                garment_part='sleeves',
                side='left'
            )

            front_right_pant_stretch_array_u, front_right_pant_stretch_array_v = set_local_stretches(
                verts=front_right_pant_verts,
                faces=front_right_pant_faces,
                design_dict=design_dict['stretches'],
                garment_part='lower'
            )
            back_right_pant_stretch_array_u, back_right_pant_stretch_array_v = set_local_stretches(
                verts=back_right_pant_verts,
                faces=back_right_pant_faces,
                design_dict=design_dict['stretches'],
                garment_part='lower'
            )
            front_left_pant_stretch_array_u, front_left_pant_stretch_array_v = set_local_stretches(
                verts=front_left_pant_verts,
                faces=front_left_pant_faces,
                design_dict=design_dict['stretches'],
                garment_part='lower'
            )
            back_left_pant_stretch_array_u, back_left_pant_stretch_array_v = set_local_stretches(
                verts=back_left_pant_verts,
                faces=back_left_pant_faces,
                design_dict=design_dict['stretches'],
                garment_part='lower'
            )

            np.savetxt(f'{mesh_set_dir}/front_shirt/stretches_u.txt', front_shirt_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/back_shirt/stretches_u.txt', back_shirt_strech_array_u)
            np.savetxt(f'{mesh_set_dir}/front_right_sleeve/stretches_u.txt', front_right_sleeve_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/back_right_sleeve/stretches_u.txt', back_right_sleeve_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/front_left_sleeve/stretches_u.txt', front_left_sleeve_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/back_left_sleeve/stretches_u.txt', back_left_sleeve_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/front_right_pant/stretches_u.txt', front_right_pant_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/back_right_pant/stretches_u.txt', back_right_pant_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/front_left_pant/stretches_u.txt', front_left_pant_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/back_left_pant/stretches_u.txt', back_left_pant_stretch_array_u)

            np.savetxt(f'{mesh_set_dir}/front_shirt/stretches_v.txt', front_shirt_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/back_shirt/stretches_v.txt', back_shirt_strech_array_v)
            np.savetxt(f'{mesh_set_dir}/front_right_sleeve/stretches_v.txt', front_right_sleeve_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/back_right_sleeve/stretches_v.txt', back_right_sleeve_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/front_left_sleeve/stretches_v.txt', front_left_sleeve_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/back_left_sleeve/stretches_v.txt', back_left_sleeve_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/front_right_pant/stretches_v.txt', front_right_pant_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/back_right_pant/stretches_v.txt', back_right_pant_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/front_left_pant/stretches_v.txt', front_left_pant_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/back_left_pant/stretches_v.txt', back_left_pant_stretch_array_v)

            np.savetxt(f'{latest_set_dir}/front_shirt/stretches_u.txt', front_shirt_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/back_shirt/stretches_u.txt', back_shirt_strech_array_u)
            np.savetxt(f'{latest_set_dir}/front_right_sleeve/stretches_u.txt', front_right_sleeve_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/back_right_sleeve/stretches_u.txt', back_right_sleeve_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/front_left_sleeve/stretches_u.txt', front_left_sleeve_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/back_left_sleeve/stretches_u.txt', back_left_sleeve_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/front_right_pant/stretches_u.txt', front_right_pant_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/back_right_pant/stretches_u.txt', back_right_pant_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/front_left_pant/stretches_u.txt', front_left_pant_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/back_left_pant/stretches_u.txt', back_left_pant_stretch_array_u)

            np.savetxt(f'{latest_set_dir}/front_shirt/stretches_v.txt', front_shirt_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/back_shirt/stretches_v.txt', back_shirt_strech_array_v)
            np.savetxt(f'{latest_set_dir}/front_right_sleeve/stretches_v.txt', front_right_sleeve_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/back_right_sleeve/stretches_v.txt', back_right_sleeve_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/front_left_sleeve/stretches_v.txt', front_left_sleeve_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/back_left_sleeve/stretches_v.txt', back_left_sleeve_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/front_right_pant/stretches_v.txt', front_right_pant_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/back_right_pant/stretches_v.txt', back_right_pant_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/front_left_pant/stretches_v.txt', front_left_pant_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/back_left_pant/stretches_v.txt', back_left_pant_stretch_array_v)

            # Store for the initial color-coded designs.
            if mesh_name == 'init' and offset_type == 'skintight':
                upper_garment_stretch_array_u, upper_garment_stretch_array_v = set_local_stretches(
                    verts=upper_garment_verts,
                    faces=upper_garment_faces,
                    design_dict=design_dict['stretches'],
                    garment_part='upper'
                )
                lower_garment_stretch_array_u, lower_garment_stretch_array_v = set_local_stretches(
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

            export(args, verts, faces, body_colors, f'{mesh_set_dir}/body-{set_element_idx:02d}', args.file_format)
            export(args, verts, faces, body_colors, f'{latest_set_dir}/body-{set_element_idx:02d}', args.file_format)
