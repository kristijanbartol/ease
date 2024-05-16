'''
Use for selecting the garment submeshes.

Relevant at the point when PBS will be applied, until then, the
3D grid creation operations can be done on the original mesh.
'''

import trimesh
import torch
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
from geometry import (
    bezier_curve,
    find_init_vertex_idx,
    project_boundaries_using_faces_deprecated,
    subdivide_mesh
)
from seams import (
    determine_pant_seams,
    determine_shirt_seams,
    determine_sleeve_seams,
    extract_parameterized_seams
)
from utils import (
    export,
    export_to_ply,
    update_color_indices
)
from mesh_sets import SETS


def select_subdivided(args, smpl_model):
    # Keep original vertices for selecting the control point locations for the Bezier curves
    orig_verts = smpl_model().vertices[0].cpu().detach().numpy()
    orig_faces = smpl_model.faces

    # Subdivide mesh for smoothness - geometric operations will be more accurate
    verts, faces = subdivide_mesh(
        verts=orig_verts, 
        faces=orig_faces
    )

    # Create trimesh.Trimesh for projecting 3D points to the mesh surface
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    num_points = 1000
    t_values = np.linspace(0, 1, num_points)

    # Iterate over body parts
    body_part_return_dict = {}
    for body_part in ['upper_front']:
        body_part_curve_points = np.empty((0, 3))
        bottom_left_point, bottom_right_point = None, None

        # Iterate over boundaries of the body part
        for boundary_key in KEYPOINTS[body_part]:
            control_point_idx_list = KEYPOINTS[body_part][boundary_key]
            control_points = [orig_verts[idx] for idx in control_point_idx_list]

            # Select Bezier control points based on the length parameter
            if boundary_key == 'left_side':
                _, bottom_left_point = extract_parameterized_seams(
                    verts=verts, 
                    garment_length=args.shirt_length, 
                    seam_vertex_indices=SEAM_IDX_DICT[body_part]['left_armpit']
                )
                control_points.append(bottom_left_point)
            if boundary_key == 'right_side':
                _, bottom_right_point = extract_parameterized_seams(
                    verts=verts, 
                    garment_length=args.shirt_length, 
                    seam_vertex_indices=SEAM_IDX_DICT[body_part]['right_armpit']
                )
                control_points.append(bottom_right_point)
            if boundary_key == 'bottom':
                mid_point = (bottom_left_point + bottom_right_point) / 2
                mid_point[2] += 0.1
                control_points.extend([bottom_left_point, mid_point, bottom_right_point])

            # Obtain Bezier curve points and concatenate to the list of unprojected boundaries
            body_part_curve_points = np.vstack([
                body_part_curve_points,
                np.array([bezier_curve(t, control_points) for t in t_values])
            ])

        # Project the Bezier curve boundaries to the mesh surface
        # TODO: Project to the triangles instead of vertices for smoother cut-out surface.
        projected_vertex_idxs = project_boundaries_using_faces_deprecated(
            mesh=mesh,
            points=body_part_curve_points
        )
        # Find the starting point on the mesh surface for the Flood Fill algorithm
        init_idx = find_init_vertex_idx(
            mesh=mesh,
            start_point=np.mean(control_points, axis=0)
        )
        # Create adjacency matrix in Garment object and apply Flood Fill
        garment = Garment(verts, faces)
        selected_verts = garment.flood_fill_vertices_subdivided(
            boundary_vertex_ids=projected_vertex_idxs, 
            starting_vertex_id=init_idx
        )
        body_part_verts, body_part_faces = garment.extract_garment_mesh(verts, faces, selected_verts, offset=0.005)
        body_part_colors = {
            'red': selected_verts
        }
        body_colors = {
            'gray': list(range(len(verts)))
        }
        body_part_garment_colors = update_color_indices(selected_verts, body_part_colors)
        export(
            body_part_verts, 
            body_part_faces, 
            body_part_garment_colors, 
            'results/tl_out/front_upper',
            args.file_format
        )
        export(
            verts, 
            faces, 
            body_colors, 
            'results/tl_out/body',
            args.file_format
        )
        body_part_return_dict[body_part] = (verts, faces, projected_vertex_idxs)
    return body_part_return_dict


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

        # Set garment mesh colors
        upper_garment_colors = {
            'red': front_v_idxs,
            'blue': back_v_idxs,
            'light_green': sleeve_indices['front_right'],
            'dark_green': sleeve_indices['back_right'],
            'brown': sleeve_indices['front_left'],
            'white': sleeve_indices['back_left']
        }
        lower_garment_colors = {
            'dark_blue': pant_indices['front_right'],
            'light_blue': pant_indices['front_left'],
            'orange': pant_indices['back_right'],
            'yellow': pant_indices['back_left']
        }

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

        # Update garment component colors
        updated_upper_garment_colors = update_color_indices(upper_indices, upper_garment_colors)
        updated_lower_garment_colors = update_color_indices(lower_indices, lower_garment_colors)

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
        export(upper_garment_verts, upper_garment_faces, updated_upper_garment_colors, f'{mesh_set_dir}/upper_garment_{mesh_name}', args.file_format)
        export(lower_garment_verts, lower_garment_faces, updated_lower_garment_colors, f'{mesh_set_dir}/lower_garment_{mesh_name}', args.file_format)

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

        export(verts, faces, body_colors, f'{mesh_set_dir}/body-{mesh_idx:2d}', args.file_format)
