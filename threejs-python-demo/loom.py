import sys
sys.path.append('/home/kristijan/LOOM/potpourri3d/src')

from enum import Enum, auto
import potpourri3d as pp3d
from smplx import SMPL
import os
from sys import platform
import torch
import numpy as np
import trimesh
from collections import deque, defaultdict
from scipy.spatial import cKDTree
import igl
import heapq
from itertools import combinations
import shutil
import json
import subprocess


def insert_midline_point(verts, faces, v_idx, front=True):
    y = verts[v_idx, 1]
    sign = 1.0 if front else -1.0
    # unique undirected edges
    edges = np.unique(
        np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
        axis=0
    )

    best = (np.inf, None, None, None)  # (score, point, i0, i1)
    for i0, i1 in edges:
        v0, v1 = verts[i0], verts[i1]
        if v0[2] * sign <= 0 or v1[2] * sign <= 0:
            continue
        t = (y - v0[1]) / (v1[1] - v0[1])
        if 0 <= t <= 1:
            # minimize how far BOTH edge endpoints are from X=0
            score = max(abs(v0[0]), abs(v1[0]))
            if score < best[0]:
                best = (score, v0 + t * (v1 - v0), i0, i1)

    _, p, i0, i1 = best
    verts_old = verts
    verts = np.vstack((verts, p[None]))
    new_idx = len(verts) - 1

    def orient(tri, n_ref):  # keep original face orientation
        a, b, c = tri
        if np.dot(np.cross(verts[b] - verts[a], verts[c] - verts[a]), n_ref) < 0:
            return [a, c, b]
        return tri

    new_faces = []
    for f in faces:
        if i0 in f and i1 in f:
            n_ref = np.cross(verts_old[f[1]] - verts_old[f[0]], verts_old[f[2]] - verts_old[f[0]])
            third = next(v for v in f if v not in (i0, i1))
            new_faces.append(orient([i0, new_idx, third], n_ref))
            new_faces.append(orient([new_idx, i1, third], n_ref))
        else:
            new_faces.append(f)

    return verts, np.asarray(new_faces, int), new_idx



REF_KPTS = {
    'upper': {
        'mid': [3168, 3500],
        'neck': [4294, 5310],
        'shoulder': [5282, 5335],
        'side': [5326, 4891],

        'sleeve': None,
        'bottom': None
    },
    'lower': {
        'side': [4164, 4303],
        'between': [1208, 1364],

        'bottom_side': None,
        'bottom_inner': None
    }
}


SHOULDER_KPT_IDX = 5335

UNIFORM_SCALE = 0.9

exclude_patch_vidxs = [
    1999,                    # left hand
    5460,                   # right hand
    408,                    # face
    3255,                   # left foot
    6736                    # right foot
]

DESIGN_TEMPLATE = {
    'upper': {
        'pos': {
            'mid': 0.1,
            'neck': 0.1,      # assym.
            'shoulder': 0.7,  # assym.
            'side': 0.3,      # assym.
        },
        'length': {
            'sleeve': 0.25,
            'bottom': 0.25,   # assym.
        },
        'flag': {
            'use_shoulder': True,
            'use_mid': False,
            'use_sleeve': True,
            'is_dress': False,
            'is_assymetric': False
        }
    },
    'lower': {
        'pos': {
            'side': 0.5,
            'between': 0.7
        },
        'length': {
            'bottom': 0.7
        },
        'flag': {
            'is_dress': False
        }
    }
}


HYPERPARAMS_TEMPLATE = {
    "stretch_coef": 2.0,
    "edges_coef": 1.0,
    "seams_coef": 0.0,
    "material_stretch_coef": 1.0,
    "seamline_strategy": "average",
    "matching_mode": "strict",
    "num_seam_iters": 1,
    "max_stretch": 0.05,
    "dart_coef": 50.0,
    "equalize_seamline_lengths": False
}


def tree():
    return defaultdict(tree)



def _extract_symm_idx(tree, kpt_pos, kpt_idx):
    if np.abs(kpt_pos[0]) < 0.01:
        return kpt_idx
    _kpt_pos = kpt_pos.copy()
    _kpt_pos[0] *= -1    # reflect across X
    _, symm_idx = tree.query(_kpt_pos)
    return symm_idx


def _extract_side_idx(mesh, idx1, idx2, z_offset: float):
    mid_x = (mesh.vertices[idx1][0] + mesh.vertices[idx2][0]) / 2.
    ref_y = mesh.vertices[idx1][1]

    query_p = np.array([mid_x, ref_y, z_offset])

    dists = np.linalg.norm(mesh.vertices - query_p, axis=1)
    closest_vertex_index = np.argmin(dists)
    return closest_vertex_index


def _side_to_full_keypoints(mesh, ref_keypoints_batch):
    tree = cKDTree(mesh.vertices)
    full_keypoints_batch = []
    for ref_keypoints in ref_keypoints_batch:
        symm_keypoints = []
        for kpt_idx in ref_keypoints:
            symm_idx = _extract_symm_idx(tree, mesh.vertices[kpt_idx], kpt_idx)
            symm_keypoints.append(symm_idx)
        full_keypoints_batch.append(ref_keypoints)
        if ref_keypoints != symm_keypoints:     # when the ref keypoints are exactly along the middle line, don't duplicate
            full_keypoints_batch.append(symm_keypoints)

    return full_keypoints_batch


def _astar_geodesic_path(mesh, start_vertex, end_vertex):
    vertices = mesh.vertices
    adjacency = mesh.vertex_neighbors

    num_vertices = len(vertices)
    visited = np.zeros(num_vertices, dtype=bool)
    previous = np.full(num_vertices, -1)
    g_score = np.full(num_vertices, np.inf)
    g_score[start_vertex] = 0

    def heuristic(v_idx):
        return np.linalg.norm(vertices[v_idx] - vertices[end_vertex])

    queue = [(heuristic(start_vertex), start_vertex)]

    while queue:
        _, current = heapq.heappop(queue)

        if visited[current]:
            continue
        visited[current] = True

        if current == end_vertex:
            break

        for neighbor in adjacency[current]:
            tentative_g = g_score[current] + np.linalg.norm(vertices[current] - vertices[neighbor])
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                previous[neighbor] = current
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(queue, (f_score, neighbor))

    # Reconstruct path
    path = []
    current = end_vertex
    while current != -1:
        path.append(current)
        current = previous[current]
    return path[::-1]


def _extract_keypoint_along_path(mesh, range_idx1, range_idx2, interp: float):
    path = _astar_geodesic_path(mesh, range_idx1, range_idx2)
    # TODO: Also implement selecting the vertex using specified distance [cm].
    return path[min(int(interp * float((len(path)))), len(path) - 1)]


def _extract_parametric_keypoint(mesh, starting_idx, dir_vector, offset, length: float):
    query_p = mesh.vertices[starting_idx] + offset + dir_vector * length
    dists = np.linalg.norm(mesh.vertices - query_p, axis=1)
    closest_vertex_index = np.argmin(dists)
    return closest_vertex_index


def _param_to_core_keypoints_upper(mesh, ref_keypoints_dict, params_dict):
    core_idx_dict = {}
    for k in ref_keypoints_dict:
        if ref_keypoints_dict[k]:
            core_idx_dict[k] = _extract_keypoint_along_path(mesh, ref_keypoints_dict[k][0], ref_keypoints_dict[k][1], params_dict[k])

    # bottom - mandatory
    core_idx_dict['bottom'] = _extract_parametric_keypoint(mesh, core_idx_dict['side'], np.array([0, -1, 0]), np.zeros(3,), params_dict['bottom'])

    # sleeves - optional
    if params_dict['sleeve'] and core_idx_dict['shoulder']:
        dir_vector = np.array([-1, 0, 0])
        sleeve_length = params_dict['sleeve']
        core_idx_dict['sleeve_up'] = _extract_parametric_keypoint(mesh, core_idx_dict['shoulder'], dir_vector, np.array([0., 0.00, 0.]), sleeve_length)
        core_idx_dict['sleeve_down'] = _extract_parametric_keypoint(mesh, core_idx_dict['side'], dir_vector, np.zeros(3,), sleeve_length)

    return core_idx_dict


def _param_to_core_keypoints_lower(mesh, ref_keypoints_dict, params_dict):
    # TODO: implement extraction of parametric keypoint on the line between two keypoints (i.e., the second keypoint should be found more robustly)
    # precisely: I should have a fixed, reference keypoint that I use to get the correct direction
    core_idx_dict = {}
    for k in ref_keypoints_dict:
        if ref_keypoints_dict[k]:
            core_idx_dict[k] = _extract_keypoint_along_path(mesh, ref_keypoints_dict[k][0], ref_keypoints_dict[k][1], params_dict[k])

    # side-bottom, but also used for inner-bottom
    inner_length = params_dict['bottom'] - (mesh.vertices[core_idx_dict['side']][1] - mesh.vertices[core_idx_dict['between']][1])
    core_idx_dict['bottom_side'] = _extract_parametric_keypoint(mesh, core_idx_dict['side'], np.array([0, -1, 0]), np.zeros(3,), length=params_dict['bottom'])
    core_idx_dict['bottom_inner'] = _extract_parametric_keypoint(mesh, core_idx_dict['between'], np.array([0, -1, 0]), np.array([-0.025, 0., 0.]), length=inner_length)

    return core_idx_dict


def _param_to_core_keypoints(mesh, ref_keypoints_dict, params_dict):
    if garment_part == 'upper':
        return _param_to_core_keypoints_upper(mesh, ref_keypoints_dict, params_dict)
    else:
        return _param_to_core_keypoints_lower(mesh, ref_keypoints_dict, params_dict)


def _core_to_side_keypoints_upper(mesh, core_idxs_dict):
    side_keypoints_batch = []

    # close sleeve boundary
    if 'sleeve_up' in core_idxs_dict:
        kpt_idx1, kpt_idx2 = core_idxs_dict['sleeve_up'], core_idxs_dict['sleeve_down']
        front_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, 0.01)
        back_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, -0.1)
        side_keypoints_batch.append([kpt_idx1, front_idx, kpt_idx2])
        side_keypoints_batch.append([kpt_idx1, back_idx,  kpt_idx2])

        side_keypoints_batch.append([core_idxs_dict['sleeve_up'], core_idxs_dict['shoulder']])
        side_keypoints_batch.append([core_idxs_dict['side'], core_idxs_dict['sleeve_down']])

    # mid-neck
    side_keypoints_batch.append([core_idxs_dict['neck'], core_idxs_dict['mid']])

    # back neck
    V, F, mid_back_idx = insert_midline_point(mesh.vertices, mesh.faces, core_idxs_dict['neck'], front=False)
    side_keypoints_batch.append([core_idxs_dict['neck'], mid_back_idx])

    # neck-shoulder
    side_keypoints_batch.append([core_idxs_dict['neck'], core_idxs_dict['shoulder']])

    # shoulder loop
    kpt_idx1, kpt_idx2 = core_idxs_dict['shoulder'], core_idxs_dict['side']
    front_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, 0.05)
    back_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, -0.1)
    side_keypoints_batch.append([kpt_idx2, front_idx, kpt_idx1])
    side_keypoints_batch.append([kpt_idx2, back_idx, kpt_idx1])

    # armpit-bottom
    side_keypoints_batch.append([core_idxs_dict['side'], core_idxs_dict['bottom']])

    # bottom
    V, F, mid_front_idx = insert_midline_point(V, F, core_idxs_dict['bottom'], front=True)
    V, F, mid_back_idx  = insert_midline_point(V, F, core_idxs_dict['bottom'], front=False)
    side_keypoints_batch.append([core_idxs_dict['bottom'], mid_front_idx])
    side_keypoints_batch.append([core_idxs_dict['bottom'], mid_back_idx])

    #V, F, _ = horizontal_plane_cut(V, F, V[core_idxs_dict['bottom']][1])

    return side_keypoints_batch, V, F


def _core_to_side_keypoints_lower(mesh, core_idxs_dict):
    side_keypoints_batch = []

    # armpit-bottom
    side_keypoints_batch.append([core_idxs_dict['side'], core_idxs_dict['bottom_side']])

    # top (waistline)
    V, F, mid_front_idx = insert_midline_point(mesh.vertices, mesh.faces, core_idxs_dict['side'], front=True)
    V, F, mid_back_idx  = insert_midline_point(V, F, core_idxs_dict['side'], front=False)
    side_keypoints_batch.append([core_idxs_dict['side'], mid_front_idx])
    side_keypoints_batch.append([mid_back_idx, core_idxs_dict['side']])

    # mid (front/back) - between
    side_keypoints_batch.append([mid_front_idx, core_idxs_dict['between']])
    side_keypoints_batch.append([mid_back_idx, core_idxs_dict['between']])

    # between-bottom
    side_keypoints_batch.append([core_idxs_dict['between'], core_idxs_dict['bottom_inner']])

    # connect bottoms
    kpt_idx1, kpt_idx2 = core_idxs_dict['bottom_side'], core_idxs_dict['bottom_inner']
    front_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, 0.02)
    back_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, -0.05)
    side_keypoints_batch.append([kpt_idx1, front_idx, kpt_idx2])
    side_keypoints_batch.append([kpt_idx1, back_idx,  kpt_idx2])

    return side_keypoints_batch, V, F

 
def _core_to_side_keypoints(mesh, core_idxs_dict):
    if garment_part == 'upper':
        return _core_to_side_keypoints_upper(mesh, core_idxs_dict)
    else:
        return _core_to_side_keypoints_lower(mesh, core_idxs_dict)


def param_to_full_keypoints(mesh, ref_keypoints_dict, params_dict):
    core_idxs_dict = _param_to_core_keypoints(mesh, ref_keypoints_dict, params_dict)
    side_keypoints_batch, newV, newF = _core_to_side_keypoints(mesh, core_idxs_dict)

    new_mesh = trimesh.Trimesh(vertices=newV, faces=newF)
    full_keypoints_batch = _side_to_full_keypoints(new_mesh, side_keypoints_batch)
    return full_keypoints_batch, new_mesh


def flood_fill_vertex_patches_with_multilabels(V, F, polylines):
    '''
    Flood fill algorithm for (multi-)labeling vertices.

    Given a set of polylines ("list of lists of vertex indices"), floods the unvisited patches.
    The unvisited patch is found by iterating through all the vertices of the mesh, checking
    whether the vertex is already visited OR whether it's a boundary vertex, and if not,
    traverses the patch in a BFS fashion.
    '''
    boundary_set = set([x for xs in polylines for x in xs])

    # The adjacency dictionary is used for faster and more convenient traversal.
    adjacency = defaultdict(set)
    for face in F:
        for i in range(3):
            vi = face[i]
            vj = face[(i + 1) % 3]
            adjacency[vi].add(vj)
            adjacency[vj].add(vi)

    patch_idxs_dict = defaultdict(set)  # for each vertex, store a set of corresponding patch labels
    current_patch_idx = 0               # start with label=0 and increment when unexplored patch is found
    excluded_patch_idxs = set()         # the excluded patches are the ones that contain excluded vertices (predefined and fixed)

    # Some vertices remain unreached by traversal, yet surrounded by already-labeled vertices (boundaries).
    # To find such vertices, we check whether all the neighboring labels are the same.
    # Afterward, these vertices are labeled using the labels of their neighbors in a separate for loop below.
    def is_surrounded(v_start):
        neighbor_patch_idxs = [patch_idxs_dict[n] for n in adjacency[v_start]]
        return all(idx == neighbor_patch_idxs[0] and len(idx) == 1 for idx in neighbor_patch_idxs)

    for v_start in range(len(V)):
        # if on the boundary, or already labeled, or "surrounded", do not process (continue)
        if v_start in boundary_set or len(patch_idxs_dict[v_start]) > 0 or is_surrounded(v_start):
            continue
        queue = deque([v_start])
        patch_idxs_dict[v_start].add(current_patch_idx)
        touched_polylines = []      # record which polylines are "touched" so that we add the corresponding idxs later
        patch_vidxs = [v_start]     # separately record patch idxs to check whether it contains the excluded idxs

        while queue:
            v = queue.popleft()
            for nbr in adjacency[v]:
                # For the boundary vertices, do not label them now. Instead, record the whole polyline as "touched".
                if nbr in boundary_set:
                    touched_polylines_idxs = []
                    for polyline_idx, polyline in enumerate(polylines):
                        if nbr in polyline:
                            touched_polylines_idxs.append(polyline_idx)
                    # However, if the boundary vertex belongs to more than one polyline, do not label as "touched".
                    # Note that this is not a problem, since the polyline will be touched at some other location.
                    if len(touched_polylines_idxs) == 1:                        
                        touched_polylines.append(polylines[touched_polylines_idxs[0]])
                    continue

                # For the "normal" (non-boundary) neighbors, label them right away and add to the queue for traversal.
                if len(patch_idxs_dict[nbr]) == 0:
                    queue.append(nbr)
                    patch_idxs_dict[nbr].add(current_patch_idx)
                    patch_vidxs.append(nbr)
                    
        # For each touched polyline, label the corresponding vertices along the polylines (with the current label).
        # Note that, when done for multiple patches (labels), the boundary vertices will "naturally" have multiple labels.
        for touched_polyline in touched_polylines:
            for tv in touched_polyline:
                patch_idxs_dict[tv].add(current_patch_idx)

        # Finally, if the excluded vertex index is part of the patch, label the whole patch as excluded.
        for excluded_vidx in exclude_patch_vidxs:
            if excluded_vidx in patch_vidxs:
                excluded_patch_idxs.add(current_patch_idx)

        current_patch_idx += 1

    # After the "regular" vertices are processed, the edge cases are the "surrounded" vertices that are now labeled.
    for v in range(len(V)):
        if len(patch_idxs_dict[v]) == 0:
            nbr = next(iter(adjacency[v]))
            patch_idxs_dict[v].add(nbr)

    return patch_idxs_dict, excluded_patch_idxs


def extract_patch(V, face_list):
    face_array = np.array(face_list)
    unique_verts, inverse_indices = np.unique(face_array.flatten(), return_inverse=True)

    V_patch = V[unique_verts]
    F_patch = inverse_indices.reshape((-1, 3))
    patch_mesh = trimesh.Trimesh(vertices=V_patch, faces=F_patch, process=False)

    return patch_mesh, unique_verts


def extract_and_save_patch_meshes(V, F, vertex_to_patch_idxs_dict, excluded_patch_idxs):
    '''
    Extract and save patches based on the flood fill vertex labels.

    In principle, the idea is to collect all the faces that belong to each patch label.
    Based on the selected faces, we find unique vertices and select the patch meshes.
    '''
    patch_faces = defaultdict(list)

    for face_idx, face in enumerate(F):
        v0, v1, v2 = face
        common_patch_idxs = set(vertex_to_patch_idxs_dict[v0]) & set(vertex_to_patch_idxs_dict[v1]) & set(vertex_to_patch_idxs_dict[v2])
        # When the face with multiple common labels is found, it certainly belongs to excluded patches (edge case).
        # In this case, we use an inner for loop and if statement to find any excluded label to use for this face.
        if len(common_patch_idxs) > 1:
            for excluded_patch_idx in excluded_patch_idxs:
                if excluded_patch_idx in common_patch_idxs:
                    patch_faces[excluded_patch_idx].append(face)
                    break
        else:
            for lbl in common_patch_idxs:
                patch_faces[lbl].append(face)

    patches = [trimesh.Trimesh()] * len(patch_faces)
    vertex_patch_index_map = dict()

    for patch_id, face_list in patch_faces.items():
        # Another edge case. This solution works for the current designs but is not general and could fail in the future.
        if len(face_list) < 20:
            excluded_patch_idxs.add(patch_id)

        patch_mesh, unique_verts = extract_patch(V, face_list)
        patches[patch_id] = patch_mesh

        # From the vertex indices from old to new, i.e., main mesh to patches.
        for local_idx, original_idx in enumerate(unique_verts):
            if original_idx not in vertex_patch_index_map:
                vertex_patch_index_map[original_idx] = {}
            vertex_patch_index_map[original_idx][patch_id] = local_idx

    # Finally, store valid patch labels for later processing.
    valid_patch_idxs = set(range(len(patch_faces))) - set(excluded_patch_idxs)
    
    return patches, patch_faces, valid_patch_idxs, vertex_patch_index_map


def extract_seamlines(boundary_indices_array, v_to_patch_idxs_dict, valid_patch_idxs, vertex_patch_index_map):
    '''
    Extract seamline indices as pairs of corresponding vertices in the neighboring patches.

    Each seamline is a separate entry and always belongs to a single pair of patches (although not vice versa).
    The boundaries (other) are on the border of the garment and connect with the excluded patches (e.g., head etc.).
    
    vertex_patch_index_map: {vidx: {label: patch_vidx}}, e.g., {115: {3: 312, 4: 1117, 6: 2}}
    seamlines_dict_list: [{(patch_idx1, patch_idx2): [(vidx_patch1, vidx_patch2)]}]
    '''
    seamlines_dict_list = []
    for boundary_indices in boundary_indices_array:
        seamlines_dict = defaultdict(list)
        is_seamline = True

        for vidx in boundary_indices:
            v_patch_idxs = v_to_patch_idxs_dict[vidx]
            filtered_patch_idxs = sorted(set(v_patch_idxs) & valid_patch_idxs)
            if len(filtered_patch_idxs) == 1:    # then it's a boundary, not a seamline
                is_seamline = False
                break
            patch_pairs = list(combinations(filtered_patch_idxs, 2))

            for patch_pair in patch_pairs:
                patch1_idx = vertex_patch_index_map[vidx][patch_pair[0]]
                patch2_idx = vertex_patch_index_map[vidx][patch_pair[1]]
                seamlines_dict[patch_pair].append((patch1_idx, patch2_idx))

        # After collecting all the seamlines, there are tips of seamlines, connected via either one or two vertices.
        # Although these are logically valid connections, they are not useful for the energy minimization.
        for patch_pair in list(seamlines_dict.keys()):
            if len(seamlines_dict[patch_pair]) <= 2:
                del(seamlines_dict[patch_pair])

        # Finally, add only the seamlines (not other boundaries).
        if is_seamline:
            seamlines_dict_list.append(seamlines_dict)
    return seamlines_dict_list


def cut_paths(V, F, keypoints_batch):
    path_solver = pp3d.ExtendedEdgeFlipGeodesicSolver(V, F)
    keypoint_coordinates = []
    for pair in keypoints_batch:
        keypoint_coordinates.append([V[pair[0]], V[pair[-1]]])
    newV, newF, boundary_indices_array = path_solver.apply_cuts(keypoints_batch, keypoint_coordinates)

    v_patch_idxs_dict, excluded_patch_idxs = flood_fill_vertex_patches_with_multilabels(newV, newF, boundary_indices_array)
    patches, patch_faces, valid_patch_idxs, vertex_patch_index_map = extract_and_save_patch_meshes(newV, newF, v_patch_idxs_dict, excluded_patch_idxs)
    seamlines_dict_list = extract_seamlines(boundary_indices_array, v_patch_idxs_dict, valid_patch_idxs, vertex_patch_index_map)

    return trimesh.Trimesh(vertices=newV, faces=newF), patches, patch_faces, seamlines_dict_list, valid_patch_idxs


def assign_patch_labels(patches, garment_part, valid_patch_idxs, ref_point):
    symm_ref_point = ref_point.copy()
    symm_ref_point[0] *= -1    # reflect across X

    # the right point is the one with the smaller X coordinate and vice versa
    ref_point_right = ref_point if ref_point[0] < symm_ref_point[0] else symm_ref_point
    ref_point_left  = ref_point if ref_point[0] > symm_ref_point[0] else symm_ref_point

    patch_labels = defaultdict(list)
    for patch_idx, patch in enumerate(patches):
        if patch_idx in valid_patch_idxs:
            # check whether the patch is part of the sleeve
            if garment_part == 'upper':
                count_right = (patch.vertices[:, 0] < ref_point_right[0]).sum()
                is_majority_right = count_right > (len(patch.vertices) / 2)
                count_left = (patch.vertices[:, 0] > ref_point_left[0]).sum()
                is_majority_left = count_left > (len(patch.vertices) / 2)

                if is_majority_right or is_majority_left:
                    patch_labels['sleeve'].append(patch_idx)
            
            # check whether the patch is a back patch
            count_back = (patch.vertices[:, 2] < ref_point[2]).sum()
            is_majority_back = count_back > (len(patch.vertices) / 2)
            if is_majority_back:
                patch_labels['back'].append(patch_idx)

    return patch_labels


def extract_target_patches(target_mesh, ref_patches, patch_faces):
    target_patches = []
    for patch_idx, ref_patch in enumerate(ref_patches):
        patch_mesh, _ = extract_patch(target_mesh.vertices, patch_faces[patch_idx])
        target_patches.append(patch_mesh)
    return target_patches


def prepare_body_meshes(gender):
    smpl_model = SMPL(
        model_path=os.path.join(SMPL_DIR, f'SMPL_{gender.upper()}.pkl'), 
        gender=gender
    )
    pose = torch.zeros((1, 23 * 3))
    orig_verts = smpl_model(
        body_pose=pose,
        betas=torch.zeros((1, 10))
    ).vertices[0].cpu().detach().numpy()

    smpl_faces = smpl_model.faces
    orig_mesh = trimesh.Trimesh(vertices=orig_verts, faces=smpl_faces)

    return smpl_model, { 'orig': orig_mesh }


def prepare_ref(design_params, orig_mesh, garment_part):
    params_dict = {k: v for key in ['pos', 'length'] for k, v in design_params[garment_part][key].items()}
    full_keypoints_batch, new_mesh = param_to_full_keypoints(orig_mesh, REF_KPTS[garment_part], params_dict)

    mesh_dir = 'data/meshes/'
    os.makedirs(mesh_dir, exist_ok=True)
    orig_mesh.export(os.path.join(mesh_dir, f'{garment_part}_orig.ply'))
    new_mesh.export(os.path.join(mesh_dir, f'{garment_part}_new.ply'))

    cut_mesh, patches, patch_faces, seamlines_dict_list, valid_patch_idxs = cut_paths(new_mesh.vertices, new_mesh.faces, full_keypoints_batch)
    patch_labels_dict = assign_patch_labels(patches, garment_part, valid_patch_idxs, orig_mesh.vertices[SHOULDER_KPT_IDX])

    return cut_mesh, patches, patch_faces, seamlines_dict_list, valid_patch_idxs, patch_labels_dict


def export_patches(patches, valid_patch_idxs, garment_part):
    part_patches_dir = f'data/patches/{garment_part}/'
    if os.path.isdir(part_patches_dir):
        shutil.rmtree(part_patches_dir)
    for patch_idx, patch in enumerate(patches):
        if patch_idx in valid_patch_idxs:
            patch_dir = f'{part_patches_dir}/patch_{patch_idx:02d}/'
            os.makedirs(patch_dir)
            for ext in ['ply', 'obj']:  # need OBJ for the flattening optimization
                patch.export(f'{patch_dir}/ref.{ext}')


def export_seamlines(seamlines_dict_list, garment_part):
    seamline_dir = f'data/seamlines/{garment_part}/'
    if os.path.isdir(seamline_dir):
        shutil.rmtree(seamline_dir)
    os.makedirs(seamline_dir)
    for seamline_idx, seamline_dict in enumerate(seamlines_dict_list):
        for patch_pair in seamline_dict:
            fpath = f'{seamline_dir}/seam-{seamline_idx}_{patch_pair[0]}-{patch_pair[1]}.txt'
            with open(fpath, mode='w') as seam_f:
                seam_f.write(f'{patch_pair[0]}\n{patch_pair[1]}\n')
                for vidx_pair in seamline_dict[patch_pair]:
                    seam_f.write(f'{vidx_pair[0]} {vidx_pair[1]}\n')


def export_scales(patches, valid_patch_idxs, garment_part):
    part_scales_dir = f'data/scales/{garment_part}/'
    if os.path.isdir(part_scales_dir):
        shutil.rmtree(part_scales_dir)
    for patch_idx, patch in enumerate(patches):
        if patch_idx in valid_patch_idxs:
            patch_scales_dir = os.path.join(part_scales_dir, f'patch_{patch_idx:02d}')
            os.makedirs(patch_scales_dir)

            fpath_u = f'{patch_scales_dir}/scales_u.txt'
            fpath_v = f'{patch_scales_dir}/scales_v.txt'

            with open(fpath_u, 'w') as f_u:
                for _ in patch.faces:
                    f_u.write(f"{UNIFORM_SCALE}\n")
            with open(fpath_v, 'w') as f_v:
                for _ in patch.faces:
                    f_v.write("1.0\n")


def export_patch_labels(patch_labels_dict, garment_part):
    part_labels_dir = f'data/labels/{garment_part}/'
    if os.path.isdir(part_labels_dir):
        shutil.rmtree(part_labels_dir)
    os.makedirs(part_labels_dir)
    for label in patch_labels_dict:
        fpath = os.path.join(part_labels_dir, f'{label}.txt')
        with open(fpath, 'w') as f:
            for patch_idx in patch_labels_dict[label]:
                f.write(f'{patch_idx} ')


def create_latest_dir(valid_patch_idxs, garment_part):
    latest_pattern_result_dir = f'results/pattern/latest/{garment_part}/'
    if os.path.isdir(latest_pattern_result_dir):
        shutil.rmtree(latest_pattern_result_dir)
    for patch_idx in valid_patch_idxs:
        os.makedirs(os.path.join(latest_pattern_result_dir, f'patch_{patch_idx}'))


def run_optimization():
    # Get absolute paths
    current_dir = os.getcwd()
    if platform == 'darwin':
        cpp_program_path = os.path.join(current_dir, "anisotropic-parameterization/cmake-build-debug/loom")
    else:
        cpp_program_path = os.path.join(current_dir, "anisotropic-parameterization/build/loom")
    root_project_path_arg = os.path.abspath(current_dir)
    
    # Ensure the executable exists
    if not os.path.exists(cpp_program_path):
        raise FileNotFoundError(f"Executable not found at: {cpp_program_path}")
    
    # Start with base command and config file
    command = [
        cpp_program_path,
        "--config", "anisotropic-parameterization/configs/default.json"
    ]
    
    # Add other parameters with their values
    param_mapping = {
        "matching_mode": "--matching-mode",
        "seamline_strategy": "--seamline-strategy",
        "num_seam_iters": "--num-seam-iters",
        "max_stretch": "--max-stretch",
        "material_stretch_coef": "--material-stretch-coef",
        "stretch_coef": "--stretch-coef",
        "edges_coef": "--edges-coef",
        "seams_coef": "--seams-coef",
        "dart_coef": "--dart-coef"
    }
    
    print(f"Running command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True,
            env=os.environ.copy()
        )
        print(f"Program output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the C++ program: {e}")
        print(f"Program output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


def process_config(config):
    design_params = tree()
    design_params['upper']['pos']['mid'] = config['mid']
    design_params['upper']['pos']['neck'] = config['neck']
    design_params['upper']['pos']['shoulder']= config['shoulder']
    design_params['upper']['pos']['side'] = config['upper_side']

    design_params['upper']['length']['sleeve'] = config['sleeve']
    design_params['upper']['length']['bottom'] = config['upper_bottom']

    design_params['lower']['pos']['side'] = config['lower_side']
    design_params['lower']['pos']['between'] = config['between']

    design_params['lower']['length']['bottom'] = config['lower_bottom']

    design_params['upper_scale'] = config['upper_scale']
    design_params['lower_scale'] = config['lower_scale']

    '''
    "stretch_coef": 2.0,
    "edges_coef": 1.0,
    "seams_coef": 100.0,
    "material_stretch_coef": 1.0,
    "seamline_strategy": "average",
    "matching_mode": "strict",
    "num_seam_iters": 1,
    "max_stretch": 0.05,
    "dart_coef": 50.0,
    "equalize_seamline_lengths": false
    '''
    hyperparams = defaultdict()
    hyperparams['stretch_coef'] = 2.0
    hyperparams['edges_coef'] = 1.0
    hyperparams['seams_coef'] = 100.0
    hyperparams['material_stretch_coef'] = 1.0
    hyperparams['seamline_strategy'] = 'average'
    hyperparams['matching_mode'] = 'strict'
    hyperparams['num_seam_iters'] = 1
    hyperparams['max_stretch'] = 0.05
    hyperparams['dart_coef'] = 50.0
    hyperparams['equalize_seamline_lengths'] = False

    # add pose & shape (pre-)selection (not individual parameters)

    return design_params, hyperparams


if __name__ == '__main__':
    '''
    if platform == 'darwin':
        PROJECT_DIR = '/Users/kristijanbartol/LOOM/'
        SMPL_DIR = '/Users/kristijanbartol/data/smpl/models/'
    else:
        PROJECT_DIR = '/home/kristijan/LOOM/'
        SMPL_DIR = '/home/kristijan/data/smpl/models/'

    with open('config/setup/loom.json') as config_f:
        config = json.load(config_f)
    experiment_name, design_params, hyperparams, body_set = process_config(config)

    is_dress = design_params['upper']['flag']['is_dress']
    is_skirt = design_params['lower']['flag']['is_skirt']

    run_design(SMPL_DIR, design_params, body_set, is_dress, is_skirt)
    run_loom_optimization(hyperparams)
    evaluate_experiment(PROJECT_DIR, SMPL_DIR, experiment_name, design_params, body_set, is_dress, is_skirt)
    '''


    if platform == 'darwin':
        PROJECT_DIR = '/Users/kristijanbartol/LOOM/'
        SMPL_DIR = '/Users/kristijanbartol/data/smpl/models/'
    else:
        PROJECT_DIR = '/home/kristijan/LOOM/'
        SMPL_DIR = '/home/kristijan/data/smpl/models/'

    GENDER = 'female'

    with open('config/setup/loom_collapsed.json') as config_f:
        config = json.load(config_f)

    #experiment_name, design_params, hyperparams, body_set = process_config(config)

    smpl_model, meshes_dict = prepare_body_meshes(gender=GENDER)
    design_params, hyperparams = process_config(config)

    for garment_part in ['upper', 'lower']:
        cut_mesh, patches, patch_faces, seamlines_dict_list, valid_patch_idxs, patch_labels_dict = prepare_ref(design_params, meshes_dict['orig'], garment_part)

        cut_mesh.export('data/meshes/body.ply')

        export_patches(patches, valid_patch_idxs, garment_part)
        export_seamlines(seamlines_dict_list, garment_part)
        export_scales(patches, valid_patch_idxs, garment_part)
        export_patch_labels(patch_labels_dict, garment_part)
        create_latest_dir(valid_patch_idxs, garment_part)

        garment_mesh = trimesh.util.concatenate([patches[idx] for idx in valid_patch_idxs])

    run_optimization()
