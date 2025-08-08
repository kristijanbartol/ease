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

from utils import insert_midline_point


class TraversalType(Enum):
    GEODESIC = auto()
    SHORTEST_PATH = auto()
    CIRCULAR = auto()
    PLANE_CUT = auto()


# NOTE: for symmetric designs, all keypoints are given on one side, otherwise, they are considered assymetric
ref_keypoints_dict1 = {
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

params1 = {
    'upper': {
        'mid': 0.7,         # [0., 1.]
        'neck': 0.1,        # [0., 1.]
        'shoulder': 0.7,    # [0., 1.]
        'side': 0.3,      # [0., 1.]

        'sleeve': 0.25,      # [m]
        'bottom': 0.25      # [m]
    },
    'lower': {
        'side': 0.5,         # [0., 1.]
        'between': 0.7,     # [0., 1.]

        'bottom': 0.6       # [m]
    }
}

'''
traversal_types1 = {
    'mid': TraversalType.GEODESIC,
    'neck': TraversalType.SHORTEST_PATH,
    'shoulder': TraversalType.CIRCULAR,
    'side': TraversalType.GEODESIC,

    'sleeve': TraversalType.GEODESIC,
    'bottom': TraversalType.PLANE_CUT
}
'''

exclude_patch_vidxs = [
    1999,                    # left hand
    5460,                   # right hand
    408,                    # face
    3255,                   # left foot
    6736                    # right foot
]



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
    #core_idx_dict['bottom_side'] = 6607
    core_idx_dict['bottom_inner'] = _extract_parametric_keypoint(mesh, core_idx_dict['between'], np.array([0, -1, 0]), np.array([-0.025, 0., 0.]), length=inner_length)
    #core_idx_dict['bottom_inner'] = 6598

    return core_idx_dict


def _param_to_core_keypoints(garment_part, mesh, ref_keypoints_dict, params_dict):
    if garment_part == 'upper':
        return _param_to_core_keypoints_upper(mesh, ref_keypoints_dict[garment_part], params_dict[garment_part])
    else:
        return _param_to_core_keypoints_lower(mesh, ref_keypoints_dict[garment_part], params_dict[garment_part])


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

 
def _core_to_side_keypoints(garment_part, mesh, core_idxs_dict):
    if garment_part == 'upper':
        return _core_to_side_keypoints_upper(mesh, core_idxs_dict)
    else:
        return _core_to_side_keypoints_lower(mesh, core_idxs_dict)


def param_to_full_keypoints(mesh, ref_keypoints_dict, params_dict, garment_part):
    core_idxs_dict = _param_to_core_keypoints(garment_part, mesh, ref_keypoints_dict, params_dict)
    side_keypoints_batch, newV, newF = _core_to_side_keypoints(garment_part, mesh, core_idxs_dict)

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

    labels_dict = defaultdict(set)  # for each vertex, store a set of corresponding patch labels
    current_label = 0               # start with label=0 and increment when unexplored patch is found
    excluded_labels = set()         # the excluded patches are the ones that contain excluded vertices (predefined and fixed)

    # Some vertices remain unreached by traversal, yet surrounded by already-labeled vertices (boundaries).
    # To find such vertices, we check whether all the neighboring labels are the same.
    # Afterward, these vertices are labeled using the labels of their neighbors in a separate for loop below.
    def is_surrounded(v_start):
        neighbor_labels = [labels_dict[n] for n in adjacency[v_start]]
        return all(lab == neighbor_labels[0] and len(lab) == 1 for lab in neighbor_labels)

    for v_start in range(len(V)):
        # if on the boundary, or already labeled, or "surrounded", do not process (continue)
        if v_start in boundary_set or len(labels_dict[v_start]) > 0 or is_surrounded(v_start):
            continue
        queue = deque([v_start])
        labels_dict[v_start].add(current_label)
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
                if len(labels_dict[nbr]) == 0:
                    queue.append(nbr)
                    labels_dict[nbr].add(current_label)
                    patch_vidxs.append(nbr)
                    
        # For each touched polyline, label the corresponding vertices along the polylines (with the current label).
        # Note that, when done for multiple patches (labels), the boundary vertices will "naturally" have multiple labels.
        for touched_polyline in touched_polylines:
            for tv in touched_polyline:
                labels_dict[tv].add(current_label)

        # Finally, if the excluded vertex index is part of the patch, label the whole patch as excluded.
        for excluded_vidx in exclude_patch_vidxs:
            if excluded_vidx in patch_vidxs:
                excluded_labels.add(current_label)

        current_label += 1

    # After the "regular" vertices are processed, the edge cases are the "surrounded" vertices that are now labeled.
    for v in range(len(V)):
        if len(labels_dict[v]) == 0:
            nbr = next(iter(adjacency[v]))
            labels_dict[v].add(nbr)

    return labels_dict, excluded_labels


def extract_and_save_patch_meshes(V, F, vertex_labels_dict, excluded_labels):
    '''
    Extract and save patches based on the flood fill vertex labels.

    In principle, the idea is to collect all the faces that belong to each patch label.
    Based on the selected faces, we find unique vertices and select the patch meshes.
    '''
    patch_faces = defaultdict(list)

    for face_idx, face in enumerate(F):
        v0, v1, v2 = face
        common_labels = set(vertex_labels_dict[v0]) & set(vertex_labels_dict[v1]) & set(vertex_labels_dict[v2])
        # When the face with multiple common labels is found, it certainly belongs to excluded patches (edge case).
        # In this case, we use an inner for loop and if statement to find any excluded label to use for this face.
        if len(common_labels) > 1:
            for excluded_label in excluded_labels:
                if excluded_label in common_labels:
                    patch_faces[excluded_label].append(face)
                    break
        else:
            for lbl in common_labels:
                patch_faces[lbl].append(face)

    patches = [trimesh.Trimesh()] * len(patch_faces)
    vertex_patch_index_map = dict()

    for patch_id, face_list in patch_faces.items():
        # Another edge case. This solution works for the current designs but is not general and could fail in the future.
        if len(face_list) < 20:
            excluded_labels.add(patch_id)

        face_array = np.array(face_list)
        unique_verts, inverse_indices = np.unique(face_array.flatten(), return_inverse=True)

        # From the vertex indices from old to new, i.e., main mesh to patches.
        for local_idx, original_idx in enumerate(unique_verts):
            if original_idx not in vertex_patch_index_map:
                vertex_patch_index_map[original_idx] = {}
            vertex_patch_index_map[original_idx][patch_id] = local_idx

        V_patch = V[unique_verts]
        F_patch = inverse_indices.reshape((-1, 3))
        mesh_patch = trimesh.Trimesh(vertices=V_patch, faces=F_patch, process=False)
        patches[patch_id] = mesh_patch

    # Finally, store valid patch labels for later processing.
    valid_patch_labels = set(range(len(patch_faces))) - set(excluded_labels)
    
    return patches, valid_patch_labels, vertex_patch_index_map


def extract_seamlines(boundary_indices_array, v_labels_dict, valid_patch_labels, vertex_patch_index_map):
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
            v_labels = v_labels_dict[vidx]
            filtered_labels = sorted(set(v_labels) & valid_patch_labels)
            if len(filtered_labels) == 1:    # then it's a boundary, not a seamline
                is_seamline = False
                break
            patch_pairs = list(combinations(filtered_labels, 2))

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

    v_labels_dict, excluded_labels = flood_fill_vertex_patches_with_multilabels(newV, newF, boundary_indices_array)
    patches, valid_patch_labels, vertex_patch_index_map = extract_and_save_patch_meshes(newV, newF, v_labels_dict, excluded_labels)
    seamlines_dict_list = extract_seamlines(boundary_indices_array, v_labels_dict, valid_patch_labels, vertex_patch_index_map)

    return trimesh.Trimesh(vertices=newV, faces=newF), patches, seamlines_dict_list, valid_patch_labels

 
def _barycentric_coordinates(P, A, B, C):
    # Compute vectors
    v0 = B - A
    v1 = C - A
    v2 = P - A

    # Compute dot products
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    # Compute denominator
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        raise ValueError("Degenerate triangle")

    # Compute barycentric coordinates
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return np.array([u, v, w])



def transfer_topology(orig_mesh, cut_mesh, target_mesh):
    # Find how many new vertices were added
    N_orig = len(orig_mesh.vertices)
    N_new = len(cut_mesh.vertices)
    assert N_new >= N_orig
    new_vertex_indices = np.arange(N_orig, N_new)

    # Get added vertices in A'
    added_vertices_Ap = cut_mesh.vertices[new_vertex_indices]

    # Use libigl to find closest triangle on A for each added vertex
    # Convert to numpy
    V_A = np.array(orig_mesh.vertices)
    F_A = np.array(orig_mesh.faces, dtype=np.int32)

    P = np.array(added_vertices_Ap)
    sq_dists, closest_faces, closest_pts = igl.point_mesh_squared_distance(P, V_A, F_A)

    # Now compute barycentric coordinates for each added vertex
    bary_coords = []
    for i, v in enumerate(P):
        f_idx = closest_faces[i]
        tri = V_A[F_A[f_idx]]
        b = _barycentric_coordinates(v, tri[0], tri[1], tri[2])
        bary_coords.append((f_idx, b))

    # Interpolate new vertex positions on mesh_B
    V_B = np.array(posed_mesh.vertices)
    F_B = np.array(posed_mesh.faces, dtype=np.int32)
    new_vertices_B = []

    for f_idx, b in bary_coords:
        tri_B = V_B[F_B[f_idx]]
        new_v = b[0]*tri_B[0] + b[1]*tri_B[1] + b[2]*tri_B[2]
        new_vertices_B.append(new_v)

    new_vertices_B = np.array(new_vertices_B)

    # Assemble full new vertex array for mesh_B'
    V_Bp = np.vstack([V_B, new_vertices_B])
    F_Bp = np.array(cut_mesh.faces, dtype=np.int32)

    # Save the result
    return trimesh.Trimesh(vertices=V_Bp, faces=F_Bp, process=False)


from loom.const import standard5_pose

if __name__ == '__main__':
    if platform == 'darwin':
        PROJECT_DIR = '/Users/kristijanbartol/LOOM/'
        SMPL_DIR = '/Users/kristijanbartol/data/smpl/models/'
    else:
        PROJECT_DIR = '/home/kristijan/LOOM/'
        SMPL_DIR = '/home/kristijan/data/smpl/models/'

    smpl_model = SMPL(
        model_path=os.path.join(SMPL_DIR, f'SMPL_FEMALE.pkl'), 
        gender='female'
    )

    for garment_part in ['upper', 'lower']:
        pose = torch.zeros((1, 23 * 3))

        orig_verts = smpl_model(
            body_pose=pose,
            betas=torch.zeros((1, 10))
        ).vertices[0].cpu().detach().numpy()
        shaped_verts = smpl_model(
            betas=torch.ones((1, 10))
        ).vertices[0].cpu().detach().numpy()
        posed_verts = smpl_model(
            betas=torch.zeros((1, 10)),
            body_pose=standard5_pose()
        ).vertices[0].cpu().detach().numpy()

        smpl_faces = smpl_model.faces
        orig_mesh = trimesh.Trimesh(vertices=orig_verts, faces=smpl_faces)
        shaped_mesh = trimesh.Trimesh(vertices=shaped_verts, faces=smpl_faces)
        posed_mesh = trimesh.Trimesh(vertices=posed_verts, faces=smpl_faces)

        full_keypoints_batch, new_mesh = param_to_full_keypoints(orig_mesh, ref_keypoints_dict1, params1, garment_part)

        mesh_dir = 'data/meshes/'
        os.makedirs(mesh_dir, exist_ok=True)
        orig_mesh.export(os.path.join(mesh_dir, f'{garment_part}_ref.ply'))
        new_mesh.export(os.path.join(mesh_dir, f'{garment_part}_new.ply'))

        cut_mesh, patches, seamlines_dict_list, excluded_patch_idxs = cut_paths(new_mesh.vertices, new_mesh.faces, full_keypoints_batch)
        
        transfer_shape = transfer_topology(orig_mesh, cut_mesh, shaped_mesh)
        transfer_pose = transfer_topology(orig_mesh, cut_mesh, posed_mesh)

        cut_mesh.export(os.path.join(mesh_dir, f'{garment_part}_cut.ply'))
        transfer_shape.export(os.path.join(mesh_dir, f'{garment_part}_target-shape.ply'))
        transfer_pose.export(os.path.join(mesh_dir, f'{garment_part}_target-pose.ply'))

        os.makedirs(f'data/patches/{garment_part}/', exist_ok=True)
        for patch_idx, patch in enumerate(patches):
            patch.export(f'data/patches/{garment_part}/patch_{patch_idx}.ply')

        os.makedirs(f'data/seamlines/{garment_part}/', exist_ok=True)
        for seamline_idx, seamline_dict in enumerate(seamlines_dict_list):
            for patch_pair in seamline_dict:
                fpath = f'data/seamlines/{garment_part}/seam-{seamline_idx}_{patch_pair[0]}-{patch_pair[1]}.txt'
                with open(fpath, mode='w') as seam_f:
                    seam_f.write(f'{patch_pair[0]}\n{patch_pair[1]}\n')
                    for vidx_pair in seamline_dict[patch_pair]:
                        seam_f.write(f'{vidx_pair[0]} {vidx_pair[1]}\n')

        os.makedirs(f'data/scales/{garment_part}/', exist_ok=True)
        for patch_id, patch in enumerate(patches):
            fpath_u = f'data/scales/{garment_part}/patch{patch_id}_u.txt'
            fpath_v = f'data/scales/{garment_part}/patch{patch_id}_v.txt'
            with open(fpath_u, 'w') as f_u:
                for _ in patch.faces:
                    f_u.write("1.0\n")
            with open(fpath_v, 'w') as f_v:
                for _ in patch.faces:
                    f_v.write("1.0\n")
