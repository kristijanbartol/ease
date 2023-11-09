from smplx import SMPL

from const import (
    INIT_LEFT_BACK_PANT,
    INIT_LEFT_FRONT_PANT,
    INIT_LEFT_SLEEVE,
    INIT_RIGHT_BACK_PANT,
    INIT_RIGHT_FRONT_PANT,
    INIT_RIGHT_SLEEVE,
    INIT_UPPER_BACK,
    INIT_UPPER_FRONT,
    PANT_LENGTH,
    SEAM_IDX_DICT,
    SHIRT_LENGTH,
    SLEEVE_LENGTH
)
from garment import Garment
from seams import determine_pant_seams, determine_shirt_seams
from utils import (
    export_to_ply,
    update_color_indices
)


if __name__ == '__main__':
    smpl_model = SMPL(model_path='/data/hierprob3d/smpl/SMPL_FEMALE.pkl')
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
    offset_distance = 0.01
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
