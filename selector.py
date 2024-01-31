import trimesh
import numpy as np

from const import (
    INIT_LEFT_BACK_PANT,
    INIT_LEFT_FRONT_PANT,
    INIT_RIGHT_BACK_PANT,
    INIT_RIGHT_FRONT_PANT,
    INIT_LEFT_SLEEVE,
    INIT_RIGHT_SLEEVE,
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
    determine_pant_seams,
    determine_shirt_seams,
    find_init_vertex_idx,
    project_points_to_nearest_faces,
    subdivide_mesh
)
from seams import extract_parameterized_seams
from utils import (
    export_to_ply,
    update_color_indices
)


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
        projected_vertex_idxs = project_points_to_nearest_faces(
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
            vertex_positions=verts, 
            boundary_vertices=projected_vertex_idxs, 
            start_vertex=init_idx
        )
        body_part_verts, body_part_faces = garment.extract_garment_mesh(verts, faces, selected_verts, offset=0.005)
        body_part_colors = {
            'red': selected_verts
        }
        body_colors = {
            'gray': list(range(len(verts)))
        }
        body_part_garment_colors = update_color_indices(selected_verts, body_part_colors)
        export_to_ply(
            body_part_verts, 
            body_part_faces, 
            body_part_garment_colors, 
            'output/front_upper'
        )
        export_to_ply(
            verts, 
            faces, 
            body_colors, 
            'output/body'
        )
        body_part_return_dict[body_part] = (verts, faces, projected_vertex_idxs)
    return body_part_return_dict


def select_original(smpl_model):
    verts = smpl_model().vertices[0].cpu().detach().numpy()
    faces = smpl_model.faces

    garment = Garment(verts, faces)

    # For the front shirt
    seam_idxs_front, y_shirt_threshold = determine_shirt_seams(verts, SHIRT_LENGTH, SEAM_IDX_DICT['upper_front'])
    front_v_idxs = garment.flood_fill_vertices(verts, seam_idxs_front, y_shirt_threshold, INIT_UPPER_FRONT)
    
    # For the back shirt
    seam_idxs_back, _ = determine_shirt_seams(verts, SHIRT_LENGTH, SEAM_IDX_DICT['upper_back'])
    back_v_idxs = garment.flood_fill_vertices(verts, seam_idxs_back, y_shirt_threshold, INIT_UPPER_BACK)

    # For the right sleeve
    right_sleeve_v_idxs = garment.select_sleeve_verts(verts, INIT_RIGHT_SLEEVE, SEAM_IDX_DICT['sleeves']['right'], SLEEVE_LENGTH, -1)

    # For the left sleeve
    left_sleeve_v_idxs = garment.select_sleeve_verts(verts, INIT_LEFT_SLEEVE, SEAM_IDX_DICT['sleeves']['left'], SLEEVE_LENGTH, 1)

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

    # Create upper and lower garment meshes
    offset_distance = 0.001
    upper_indices = front_v_idxs + back_v_idxs + right_sleeve_v_idxs + left_sleeve_v_idxs
    lower_indices = sum(pant_indices.values(), [])
    upper_garment_verts, upper_garment_faces = garment.extract_garment_mesh(verts, faces, upper_indices, offset=offset_distance)
    lower_garment_verts, lower_garment_faces = garment.extract_garment_mesh(verts, faces, lower_indices, offset=offset_distance)

    # Export garment meshes with color
    upper_garment_colors = {
        'red': front_v_idxs,
        'blue': back_v_idxs,
        'light_green': right_sleeve_v_idxs,
        'dark_green': left_sleeve_v_idxs
    }
    lower_garment_colors = {
        'dark_blue': pant_indices['front_right'],
        'light_blue': pant_indices['front_left'],
        'orange': pant_indices['back_right'],
        'yellow': pant_indices['back_left']
    }
    body_colors = {
        'gray': list(range(len(verts)))
    }
    updated_upper_garment_colors = update_color_indices(upper_indices, upper_garment_colors)
    updated_lower_garment_colors = update_color_indices(lower_indices, lower_garment_colors)
    export_to_ply(upper_garment_verts, upper_garment_faces, updated_upper_garment_colors, 'output/upper_garment')
    export_to_ply(lower_garment_verts, lower_garment_faces, updated_lower_garment_colors, 'output/lower_garment')
    export_to_ply(verts, faces, body_colors, 'output/body')
