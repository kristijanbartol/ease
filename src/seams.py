import numpy as np

from src.const import SEAM_IDX_DICT


def extract_parameterized_seams(verts, garment_length, seam_vertex_indices, pant_offset=None):
    num_offset_verts = 0
    if pant_offset is not None:
        remaining_length = pant_offset
        for i in range(len(seam_vertex_indices) - 1):
            start_vertex = verts[seam_vertex_indices[i]]
            end_vertex = verts[seam_vertex_indices[i + 1]]
            
            edge_length = np.linalg.norm(end_vertex - start_vertex)

            if edge_length < remaining_length:
                remaining_length -= edge_length
                num_offset_verts += 1
            else:
                first_point = verts[seam_vertex_indices[num_offset_verts]]
                break
    else:
        first_point = None

    remaining_length = garment_length
    last_point = None

    for i in range(num_offset_verts, len(seam_vertex_indices) - 1):
        start_vertex = verts[seam_vertex_indices[i]]
        end_vertex = verts[seam_vertex_indices[i + 1]]
        
        edge_length = np.linalg.norm(end_vertex - start_vertex)

        if edge_length < remaining_length:
            remaining_length -= edge_length
            if i == len(seam_vertex_indices) - 2:   # in case we came to the end
                return seam_vertex_indices, verts[seam_vertex_indices[-1]], first_point
        else:
            # Find the point along the edge that corresponds to the remaining length
            direction = (end_vertex - start_vertex) / edge_length
            last_point = start_vertex + direction * remaining_length
            if edge_length == remaining_length:
                return seam_vertex_indices[num_offset_verts:(i + 1) + 1], last_point, first_point
            else:
                return seam_vertex_indices[num_offset_verts:(i + 1) + 2], last_point, first_point


def determine_shirt_seams(verts, shirt_length, seam_idx_dict):
    right_armpit_points, last_right_point, _ = extract_parameterized_seams(verts, shirt_length, seam_idx_dict['right_armpit'])
    left_armpit_points, _, _ = extract_parameterized_seams(verts, shirt_length, seam_idx_dict['left_armpit'])
    shirt_seams = right_armpit_points + \
                  seam_idx_dict['right_arm'] + \
                  seam_idx_dict['right_shoulder'] + \
                  seam_idx_dict['neck'] + \
                  seam_idx_dict['left_shoulder'] + \
                  seam_idx_dict['left_arm'] + \
                  left_armpit_points
    return shirt_seams, last_right_point[1]


def determine_pant_seams(verts, pant_length, seam_idx_dict, side, pant_offset, inner_seams=True):
    pant_offset = None if pant_offset < 0 else pant_offset
    pant_seams = []
    outer_points, last_outer_point, first_point = extract_parameterized_seams(verts, pant_length, seam_idx_dict[f'{side}_outer'], pant_offset)
    pant_seams += outer_points
    if inner_seams:
        inner_points, _, _ = extract_parameterized_seams(verts, pant_length, seam_idx_dict[f'{side}_inner'], pant_offset)
        pant_seams += inner_points + seam_idx_dict['mid_inner']
    if pant_offset is None:
        pant_seams += seam_idx_dict['waistline']    # TODO: Instead of waistline, cut the mesh and define y threshold
    first_point_y = first_point[1] if first_point is not None else None
    return pant_seams, last_outer_point[1], first_point_y


def determine_sleeve_seams(verts, sleeve_length, seam_idx_dict):
    up_points, last_up_point, _ = extract_parameterized_seams(verts, sleeve_length, seam_idx_dict['up'])
    down_points, last_down_point, _ = extract_parameterized_seams(verts, sleeve_length, seam_idx_dict['down'])
    sleeve_seams = up_points + down_points + seam_idx_dict['side']
    return sleeve_seams, last_up_point[0]   # return last x axis


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
