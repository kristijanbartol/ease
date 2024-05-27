import numpy as np


def extract_parameterized_seams(verts, garment_length, seam_vertex_indices, inner_seams=False):
    remaining_length = garment_length
    last_point = None

    for i in range(len(seam_vertex_indices) - 1):
        start_vertex = verts[seam_vertex_indices[i]]
        end_vertex = verts[seam_vertex_indices[i + 1]]
        
        edge_length = np.linalg.norm(end_vertex - start_vertex)

        if edge_length < remaining_length:
            remaining_length -= edge_length
        elif edge_length == remaining_length:
            # Find the point along the edge that corresponds to the remaining length
            direction = (end_vertex - start_vertex) / edge_length
            last_point = start_vertex + direction * remaining_length
            return seam_vertex_indices[:(i + 1) + 1], last_point
        else:
            return seam_vertex_indices[:(i + 1) + 2], last_point


def determine_shirt_seams(verts, shirt_length, seam_idx_dict):
    right_armpit_points, last_right_point = extract_parameterized_seams(verts, shirt_length, seam_idx_dict['right_armpit'])
    left_armpit_points, _ = extract_parameterized_seams(verts, shirt_length, seam_idx_dict['left_armpit'])
    shirt_seams = right_armpit_points + \
                  seam_idx_dict['right_arm'] + \
                  seam_idx_dict['right_shoulder'] + \
                  seam_idx_dict['neck'] + \
                  seam_idx_dict['left_shoulder'] + \
                  seam_idx_dict['left_arm'] + \
                  left_armpit_points
    return shirt_seams, last_right_point[1]


def determine_pant_seams(verts, pant_length, seam_idx_dict, side, inner_seams=True):
    pant_seams = []
    outer_points, last_outer_point = extract_parameterized_seams(verts, pant_length, seam_idx_dict[f'{side}_outer'])
    pant_seams += outer_points
    if inner_seams:
        inner_points, _ = extract_parameterized_seams(verts, pant_length, seam_idx_dict[f'{side}_inner'])
        pant_seams += inner_points + seam_idx_dict['mid_inner']
    pant_seams += seam_idx_dict['waistline']
    return pant_seams, last_outer_point[1]


def determine_sleeve_seams(verts, sleeve_length, seam_idx_dict):
    up_points, last_up_point = extract_parameterized_seams(verts, sleeve_length, seam_idx_dict['up'])
    down_points, last_down_point = extract_parameterized_seams(verts, sleeve_length, seam_idx_dict['down'])
    sleeve_seams = up_points + down_points + seam_idx_dict['side']
    return sleeve_seams, last_up_point[0]   # return last x axis
