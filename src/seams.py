import numpy as np


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
        first_point = verts[seam_vertex_indices[0]]

    remaining_length = garment_length
    last_point = None

    for i in range(num_offset_verts, len(seam_vertex_indices) - 1):
        start_vertex = verts[seam_vertex_indices[i]]
        end_vertex = verts[seam_vertex_indices[i + 1]]
        
        edge_length = np.linalg.norm(end_vertex - start_vertex)

        if edge_length < remaining_length:
            remaining_length -= edge_length
            if i == len(seam_vertex_indices) - 2:   # in case we came to the end
                return seam_vertex_indices, verts[seam_vertex_indices[-1]]
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
    return pant_seams, last_outer_point[1], first_point[1]


def determine_sleeve_seams(verts, sleeve_length, seam_idx_dict):
    up_points, last_up_point, _ = extract_parameterized_seams(verts, sleeve_length, seam_idx_dict['up'])
    down_points, last_down_point, _ = extract_parameterized_seams(verts, sleeve_length, seam_idx_dict['down'])
    sleeve_seams = up_points + down_points + seam_idx_dict['side']
    return sleeve_seams, last_up_point[0]   # return last x axis
