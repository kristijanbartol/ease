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

from loom.const import standard5_pose
import loom_core.params as params
from loom_core.geometry import insert_midline_point, find_midline_point
from loom.submodules import run_loom_optimization


class TraversalType(Enum):
    GEODESIC = auto()
    SHORTEST_PATH = auto()
    CIRCULAR = auto()
    PLANE_CUT = auto()


SHOULDER_KPT_IDX = 5335

UNIFORM_SCALE = 0.9

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
    348,                    # face
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


def _get_closest_idx(verts, query_p):
    dists = np.linalg.norm(verts - query_p, axis=1)
    closest_vertex_index = np.argmin(dists)
    return closest_vertex_index


def _extract_side_idx(mesh, idx1, idx2, z_offset: float):
    mid_x = (mesh.vertices[idx1][0] + mesh.vertices[idx2][0]) / 2.
    ref_y = mesh.vertices[idx1][1]
    query_p = np.array([mid_x, ref_y, z_offset])
    return _get_closest_idx(mesh.vertices, query_p)


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


def midline_path(mesh, start, end):
    """
    V: (N,3) float array of vertices
    F: (M,3) int array of triangle indices
    start, end: vertex indices
    """
    if start == end: return [start]
    n = len(mesh.vertices)
    x = np.abs(mesh.vertices[:, 0])

    # build adjacency
    nbr = [set() for _ in range(n)]
    for a, b, c in mesh.faces:
        nbr[a].update((b, c)); nbr[b].update((a, c)); nbr[c].update((a, b))

    # Dijkstra with "stay near X=0" cost: edge weight = average |x|
    INF = float('inf')
    dist, prev = [INF]*n, [-1]*n
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == end: break
        if d != dist[u]: continue
        for v in nbr[u]:
            w = 0.5*(x[u] + x[v])
            nd = d + w
            if nd < dist[v]:
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))

    # reconstruct path
    if prev[end] == -1: return []
    path = []
    u = end
    while u != -1:
        path.append(u)
        u = prev[u]
    return path[::-1]


def _extract_keypoint_along_path(mesh, range_idx1, range_idx2, interp: float, shortest=True):
    if shortest:
        path = _astar_geodesic_path(mesh, range_idx1, range_idx2)
    else:
        path = midline_path(mesh, range_idx1, range_idx2)
    # TODO: Also implement selecting the vertex using specified distance [cm].
    return path[min(int(interp * float((len(path)))), len(path) - 1)], path


def _extract_parametric_keypoint(mesh, starting_idx, direction_label, length: float, offset=None):
    if type(direction_label) == int:
        end_idx = direction_label
        dir_vector = mesh.vertices[end_idx] - mesh.vertices[starting_idx]
    else:   # np.ndarray
        dir_vector = direction_label
    if offset is not None:
        query_p = mesh.vertices[starting_idx] + offset + dir_vector * length
    else:
        query_p = mesh.vertices[starting_idx] + dir_vector * length
    return _get_closest_idx(mesh.vertices, query_p)


def _param_to_core_keypoints_upper(mesh, ref_keypoints_dict, params_dict):
    core_idx_dict = {}
    for k in ref_keypoints_dict:
        if ref_keypoints_dict[k] and k in params_dict:
            core_idx_dict[k], _ = _extract_keypoint_along_path(mesh, ref_keypoints_dict[k][0], ref_keypoints_dict[k][1], params_dict[k])

    # bottom - mandatory
    core_idx_dict['bottom'] = _extract_parametric_keypoint(mesh, core_idx_dict['side'], np.array([0, -1, 0]), params_dict['bottom'])

    # sleeves - optional
    if 'sleeve' in params_dict and 'shoulder' in core_idx_dict:
        dir_vector = np.array([-1, 0, 0])
        sleeve_length = params_dict['sleeve']
        core_idx_dict['sleeve_up'] = _extract_parametric_keypoint(mesh, core_idx_dict['shoulder'], dir_vector, sleeve_length)
        core_idx_dict['sleeve_down'] = _extract_parametric_keypoint(mesh, core_idx_dict['side'], dir_vector, sleeve_length)

    return core_idx_dict


def _get_same_height_idx(verts, start_idx, end_idx, height_ref_idx):
    start = verts[start_idx]
    end = verts[end_idx]
    direction = end - start
    t = (verts[height_ref_idx][1] - start[1]) / direction[1]
    point_at_y = start + t * direction
    return _get_closest_idx(verts, point_at_y)


def _param_to_core_keypoints_lower(mesh, ref_keypoints_dict, params_dict, is_skirtified):
    core_idx_dict = {}
    for k in ref_keypoints_dict:
        if ref_keypoints_dict[k] and type(ref_keypoints_dict[k]) == list and k in params_dict:
            core_idx_dict[k], _ = _extract_keypoint_along_path(mesh, ref_keypoints_dict[k][0], ref_keypoints_dict[k][1], params_dict[k])

    # side-bottom, but also used for inner-bottom
    core_idx_dict['bottom_side'] = _extract_parametric_keypoint(mesh, core_idx_dict['side'], ref_keypoints_dict['bottom_side_ref'], length=params_dict['bottom'], offset=np.array([-0.01, 0, 0]))
    if not is_skirtified:
        core_idx_dict['bottom_inner'] = _get_same_height_idx(mesh.vertices, core_idx_dict['between'], ref_keypoints_dict['bottom_inner_ref'], core_idx_dict['bottom_side'])

    return core_idx_dict


def _param_to_core_keypoints(mesh, ref_keypoints_dict, params_dict, garment_part, is_skirtified):
    if garment_part == 'upper':
        return _param_to_core_keypoints_upper(mesh, ref_keypoints_dict, params_dict)
    else:
        return _param_to_core_keypoints_lower(mesh, ref_keypoints_dict, params_dict, is_skirtified)


def _core_to_side_keypoints_upper(mesh, core_idxs_dict):
    side_keypoints_batch = []
    smpl_traversal_pairs = []

    # close sleeve boundary
    if 'sleeve_up' in core_idxs_dict:
        kpt_idx1, kpt_idx2 = core_idxs_dict['sleeve_up'], core_idxs_dict['sleeve_down']
        front_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, 0.01)
        back_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, -0.1)
        side_keypoints_batch.append([kpt_idx1, front_idx, kpt_idx2])
        side_keypoints_batch.append([kpt_idx1, back_idx,  kpt_idx2])

        side_keypoints_batch.append([core_idxs_dict['sleeve_up'], core_idxs_dict['shoulder']])
        side_keypoints_batch.append([core_idxs_dict['side'], core_idxs_dict['sleeve_down']])

    V, F = mesh.vertices, mesh.faces

    if 'shoulder' in core_idxs_dict:
        if 'head' in core_idxs_dict:
            side_keypoints_batch.append([core_idxs_dict['neck'], core_idxs_dict['head']])
            #side_keypoints_batch.append([mid_back_idx, core_idxs_dict['head']])    # normally hood is made of two pieces, but we skip it for now

        # mid-neck
        side_keypoints_batch.append([core_idxs_dict['neck'], core_idxs_dict['mid']])
        
        #V, F, mid_back_idx = insert_midline_point(mesh.vertices, mesh.faces, core_idxs_dict['neck'], front=False)
        bottom_mid_back_idx = find_midline_point(mesh.vertices, mesh.faces, core_idxs_dict['neck'], front=False)
        side_keypoints_batch.append([core_idxs_dict['neck'], bottom_mid_back_idx])

        # neck-shoulder
        side_keypoints_batch.append([core_idxs_dict['neck'], core_idxs_dict['shoulder']])

        # shoulder loop
        kpt_idx1, kpt_idx2 = core_idxs_dict['shoulder'], core_idxs_dict['side']
        front_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, 0.02)
        back_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, -0.08)
        side_keypoints_batch.append([kpt_idx2, front_idx, kpt_idx1])
        side_keypoints_batch.append([kpt_idx2, back_idx, kpt_idx1])
    else:
        side_mid_front_idx = find_midline_point(V, F, core_idxs_dict['side'], front=True)
        side_mid_back_idx = find_midline_point(V, F, core_idxs_dict['side'], front=False)
        side_keypoints_batch.append([core_idxs_dict['side'], side_mid_front_idx])
        side_keypoints_batch.append([core_idxs_dict['side'], side_mid_back_idx])

    # armpit-bottom
    side_keypoints_batch.append([core_idxs_dict['side'], core_idxs_dict['bottom']])

    # bottom
    #V, F, mid_front_idx = insert_midline_point(V, F, core_idxs_dict['bottom'], front=True)
    bottom_mid_front_idx = find_midline_point(V, F, core_idxs_dict['bottom'], front=True)
    #V, F, mid_back_idx  = insert_midline_point(V, F, core_idxs_dict['bottom'], front=False)
    bottom_mid_back_idx  = find_midline_point(V, F, core_idxs_dict['bottom'], front=False)
    side_keypoints_batch.append([core_idxs_dict['bottom'], bottom_mid_front_idx])
    side_keypoints_batch.append([core_idxs_dict['bottom'], bottom_mid_back_idx])

    #V, F, _ = horizontal_plane_cut(V, F, V[core_idxs_dict['bottom']][1])

    #return side_keypoints_batch, smpl_traversal_pairs, V, F
    return side_keypoints_batch, smpl_traversal_pairs


def _core_to_side_keypoints_lower(mesh, core_idxs_dict, is_skirtified):
    side_keypoints_batch = []
    smpl_traversal_pairs = []
    V, F = mesh.vertices, mesh.faces

    if is_skirtified:
        mid_front_idx = find_midline_point(mesh.vertices, mesh.faces, core_idxs_dict['side'], front=True)
        mid_back_idx  = find_midline_point(V, F, core_idxs_dict['side'], front=False)

        side_keypoints_batch.append([core_idxs_dict['side'], mid_front_idx])
        side_keypoints_batch.append([mid_back_idx, core_idxs_dict['side']])
        side_keypoints_batch.append([core_idxs_dict['bottom_side'], core_idxs_dict['side']])

        bottom_mid_front_idx = find_midline_point(mesh.vertices, mesh.faces, core_idxs_dict['bottom_side'], front=True)
        bottom_mid_back_idx  = find_midline_point(V, F, core_idxs_dict['bottom_side'], front=False)

        side_keypoints_batch.append([core_idxs_dict['bottom_side'], bottom_mid_front_idx])
        side_keypoints_batch.append([bottom_mid_back_idx, core_idxs_dict['bottom_side']])

    else:
        '''
        # side-bottom
        side_keypoints_batch.append([core_idxs_dict['side'], core_idxs_dict['bottom_side']])

        # top (waistline)
        mid_front_idx = find_midline_point(V, F, core_idxs_dict['side'], front=True)
        mid_back_idx  = find_midline_point(V, F, core_idxs_dict['side'], front=False)
        side_keypoints_batch.append([core_idxs_dict['side'], mid_front_idx])
        side_keypoints_batch.append([mid_back_idx, core_idxs_dict['side']])

        # mid (front/back) - between
        side_keypoints_batch.append([mid_front_idx, core_idxs_dict['between']])
        side_keypoints_batch.append([mid_back_idx, core_idxs_dict['between']])

        # between-bottom
        side_keypoints_batch.append([core_idxs_dict['between'], core_idxs_dict['bottom_inner']])

        # connect bottoms
        kpt_idx1, kpt_idx2 = core_idxs_dict['bottom_side'], core_idxs_dict['bottom_inner']
        front_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, 0.08)
        back_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, -0.14)
        side_keypoints_batch.append([kpt_idx1, front_idx, kpt_idx2])
        side_keypoints_batch.append([kpt_idx1, back_idx,  kpt_idx2])
        '''

        # FOR SUBJECT PROCESSING
        # armpit-bottom
        # TODO: implement this properly
        #       # option 1: when pants are longer than knee area, then use additional keypoint
        #       # option 2 (in addition to option 1): use Bezier curves as seam definition
        #side_keypoints_batch.append([core_idxs_dict['side'], core_idxs_dict['bottom_side']])



        side_keypoints_batch.append([core_idxs_dict['side'], 4495])
        side_keypoints_batch.append([4495, 4581])
        side_keypoints_batch.append([4581, core_idxs_dict['bottom_side']])


        #side_keypoints_batch.append([core_idxs_dict['side'], 4495])
        #side_keypoints_batch.append([4495, 4583])
        #side_keypoints_batch.append([4583, core_idxs_dict['bottom_side']])





        # top (waistline)
        
        #V, F, mid_front_idx = insert_midline_point(mesh.vertices, mesh.faces, core_idxs_dict['side'], front=True)
        mid_front_idx = find_midline_point(V, F, core_idxs_dict['side'], front=True)
        #V, F, mid_back_idx  = insert_midline_point(V, F, core_idxs_dict['side'], front=False)
        mid_back_idx  = find_midline_point(V, F, core_idxs_dict['side'], front=False)
        side_keypoints_batch.append([core_idxs_dict['side'], mid_front_idx])
        side_keypoints_batch.append([mid_back_idx, core_idxs_dict['side']])

        # mid (front/back) - between
        #smpl_traversal_pairs.append([mid_front_idx, core_idxs_dict['between']])   # traverse mid-between (front) directly on SMPL (very relevant for the male model)
        #smpl_traversal_pairs.append([mid_back_idx, core_idxs_dict['between']])   # traverse mid-between (back) directly on SMPL (very relevant for the male model)
        #side_keypoints_batch.append([mid_front_idx, core_idxs_dict['between']])



        side_keypoints_batch.append([mid_front_idx, 3145])
        side_keypoints_batch.append([3145, 3149])
        side_keypoints_batch.append([3149, 1208])
        side_keypoints_batch.append([1208, core_idxs_dict['between']])
        side_keypoints_batch.append([mid_back_idx, core_idxs_dict['between']])




        # between-bottom
        # TODO: implement this properly
        #       # option 1: when pants are longer than knee area, then use additional keypoint
        #       # option 2 (in addition to option 1): use Bezier curves as seam definition
        #side_keypoints_batch.append([core_idxs_dict['between'], core_idxs_dict['bottom_inner']])
        side_keypoints_batch.append([core_idxs_dict['between'], 4634])
        side_keypoints_batch.append([4634, 4572])
        side_keypoints_batch.append([4572, core_idxs_dict['bottom_inner']])

        # connect bottoms
        kpt_idx1, kpt_idx2 = core_idxs_dict['bottom_side'], core_idxs_dict['bottom_inner']
        front_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, 0.04)
        back_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, -0.1)
        side_keypoints_batch.append([kpt_idx1, front_idx, kpt_idx2])
        side_keypoints_batch.append([kpt_idx1, back_idx,  kpt_idx2])

        #return side_keypoints_batch, smpl_traversal_pairs, V, F
    return side_keypoints_batch, smpl_traversal_pairs

 
def _core_to_side_keypoints(mesh, core_idxs_dict, garment_part, is_skirtified):
    if garment_part == 'upper':
        return _core_to_side_keypoints_upper(mesh, core_idxs_dict)
    else:
        return _core_to_side_keypoints_lower(mesh, core_idxs_dict, is_skirtified)


def param_to_full_keypoints(t_pose_mesh, ref_keypoints_dict, params_dict, garment_part, is_skirtified):
    core_idxs_dict = _param_to_core_keypoints(t_pose_mesh, ref_keypoints_dict, params_dict, garment_part, is_skirtified)
    #side_keypoints_batch, smpl_traversal_pairs, newV, newF = _core_to_side_keypoints(t_pose_mesh, core_idxs_dict)
    side_keypoints_batch, smpl_traversal_pairs = _core_to_side_keypoints(t_pose_mesh, core_idxs_dict, garment_part, is_skirtified)

    #new_mesh = trimesh.Trimesh(vertices=newV, faces=newF)
    #full_keypoints_batch = _side_to_full_keypoints(new_mesh, side_keypoints_batch)
    full_keypoints_batch = _side_to_full_keypoints(t_pose_mesh, side_keypoints_batch)
    return full_keypoints_batch, smpl_traversal_pairs


def flood_fill_vertex_patches_with_multilabels(mesh, polylines):
    '''
    Flood fill algorithm for (multi-)labeling vertices.

    Given a set of polylines ("list of lists of vertex indices"), floods the unvisited patches.
    The unvisited patch is found by iterating through all the vertices of the mesh, checking
    whether the vertex is already visited OR whether it's a boundary vertex, and if not,
    traverses the patch in a BFS fashion.
    '''
    boundary_set = set([x for xs in polylines for x in xs])
    V, F = mesh.vertices, mesh.faces

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


def extract_and_save_patch_meshes(mesh, vertex_to_patch_idxs_dict, excluded_patch_idxs):
    '''
    Extract and save patches based on the flood fill vertex labels.

    In principle, the idea is to collect all the faces that belong to each patch label.
    Based on the selected faces, we find unique vertices and select the patch meshes.
    '''
    V, F = mesh.vertices, mesh.faces
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


def extract_seamlines(patches, boundary_indices_array, v_to_patch_idxs_dict, valid_patch_idxs, vertex_patch_index_map):
    '''
    Extract seamline indices as pairs of corresponding vertices in the neighboring patches.

    Each seamline is a separate entry and always belongs to a single pair of patches (although not vice versa).
    The boundaries (other) are on the border of the garment and connect with the excluded patches (e.g., face etc.).
    
    vertex_patch_index_map: {vidx: {label: patch_vidx}}, e.g., {115: {3: 312, 4: 1117, 6: 2}}
    seamlines_dict_list: [{(patch_idx1, patch_idx2): [(vidx_patch1, vidx_patch2)]}]
    '''
    seamlines_dict_list = []
    for boundary_indices in boundary_indices_array:
        seamlines_dict = defaultdict(list)
        is_seamline = True

        for vidx in boundary_indices:
            v_patch_idxs = v_to_patch_idxs_dict[vidx]
            #filtered_patch_idxs = sorted(set(v_patch_idxs) & valid_patch_idxs)
            filtered_patch_idxs = sorted(set(v_patch_idxs) & valid_patch_idxs & set(vertex_patch_index_map[vidx].keys()))
            #if len(filtered_patch_idxs) == 1:    # then it's a boundary, not a seamline
            if len(filtered_patch_idxs) == 0:    # then it's a boundary, not a seamline
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
            if len(seamlines_dict[patch_pair]) <= 1:
                del(seamlines_dict[patch_pair])

        # Finally, add only the seamlines (not other boundaries).
        if is_seamline and len(seamlines_dict) > 0:
            seamlines_dict_list.append(seamlines_dict)
        
    symmetric_seamline_flags = [False] * len(seamlines_dict_list)
    for seam_idx in range(len(seamlines_dict_list)):
        patch_pair, vertex_pairs_list = next(iter(seamlines_dict_list[seam_idx].items()))    # take the one and only seam (using dictionary to have the label pair (patch1_idx, patch2_idx))
        patch_element = 0   # use the first patch for fetching the vertex indices and using the patch idx
        patch_idx = patch_pair[patch_element]
        verts = np.array([patches[patch_idx].vertices[vertex_pair[patch_element]] for vertex_pair in vertex_pairs_list])
        if np.mean(np.abs(verts[:, 0]) < 9e-3) > 0.9:
            symmetric_seamline_flags[seam_idx] = True

    return seamlines_dict_list, symmetric_seamline_flags


def cut_paths(template_mesh, ref_mesh, keypoints_batch, smpl_traversal_pairs):
    path_solver = pp3d.ExtendedEdgeFlipGeodesicSolver(template_mesh.vertices, template_mesh.faces)

    keypoint_coordinates = []
    for kpts in keypoints_batch:
        keypoint_coordinates.append([template_mesh.vertices[kpts[0]], template_mesh.vertices[kpts[-1]]])

    cutV, cutF, boundary_indices_array = path_solver.apply_cuts(keypoints_batch, keypoint_coordinates)
    cut_mesh = trimesh.Trimesh(vertices=cutV, faces=cutF)
    cut_mesh.export('cut_mesh.ply')

    if not np.allclose(template_mesh.vertices, ref_mesh.vertices):
        ref_mesh_with_cuts = transfer_topology(template_mesh, cut_mesh, ref_mesh)
    else:
        ref_mesh_with_cuts = cut_mesh

    v_patch_idxs_dict, excluded_patch_idxs = flood_fill_vertex_patches_with_multilabels(cut_mesh, boundary_indices_array)
    patches, patch_faces, valid_patch_idxs, vertex_patch_index_map = extract_and_save_patch_meshes(ref_mesh_with_cuts, v_patch_idxs_dict, excluded_patch_idxs)

    seamlines_dict_list, symmetric_seamline_flags = extract_seamlines(patches, boundary_indices_array, v_patch_idxs_dict, valid_patch_idxs, vertex_patch_index_map)

    return trimesh.Trimesh(vertices=cutV, faces=cutF), patches, patch_faces, seamlines_dict_list, symmetric_seamline_flags, valid_patch_idxs


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
    V_B = np.array(target_mesh.vertices)
    F_B = np.array(target_mesh.faces, dtype=np.int32)
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


def prepare_body_meshes(smpl_dir, body_set, is_skirtified=False):
    gender = body_set['genders'][0]
    target_poses = [getattr(params, body_set['poses'][idx]) for idx in range(1, len(body_set['poses']))]

    if is_skirtified:
        skirtified_dirpath = f'data/skirtified/{body_set["name"]}/'

        zero_mesh = trimesh.load(os.path.join(skirtified_dirpath, 'zero.ply'))
        template_mesh = trimesh.load(os.path.join(skirtified_dirpath, 'template.ply'))
        ref_mesh = trimesh.load(os.path.join(skirtified_dirpath, 'ref.ply'))
        target_meshes = []
        if len(target_poses) > 0:
            target_meshes.append(trimesh.load(os.path.join(skirtified_dirpath, 'target.ply')))

    else:
        smpl_model = SMPL(
            model_path=os.path.join(smpl_dir, f'SMPL_{gender.upper()}.pkl'), 
            gender=gender
        )
        pose = getattr(params, body_set['poses'][0])()
        betas = getattr(params, body_set['shapes'][0])()

        zero_verts = smpl_model(
            body_pose=params.t_pose(),
            betas=betas
        ).vertices[0].cpu().detach().numpy()
        template_verts = smpl_model(
            body_pose=params.t_pose_for_cutting(),
            betas=betas
        ).vertices[0].cpu().detach().numpy()
        ref_verts = smpl_model(
            body_pose=pose,
            betas=betas
        ).vertices[0].cpu().detach().numpy()

        target_verts_list = []
        for target_pose in target_poses:
            posed_verts = smpl_model(
                betas=torch.zeros((1, 10)),
                body_pose=target_poses[0]()
            ).vertices[0].cpu().detach().numpy()
            target_verts_list.append(posed_verts)

        smpl_faces = smpl_model.faces

        zero_mesh = trimesh.Trimesh(vertices=zero_verts, faces=smpl_faces)
        template_mesh = trimesh.Trimesh(vertices=template_verts, faces=smpl_faces)
        ref_mesh = trimesh.Trimesh(vertices=ref_verts, faces=smpl_faces)
        target_meshes = [trimesh.Trimesh(vertices=target_verts, faces=smpl_faces) for target_verts in target_verts_list]

        os.makedirs('data/meshes', exist_ok=True)
        zero_mesh.export('data/meshes/zero.ply')
        template_mesh.export('data/meshes/template.ply')
        ref_mesh.export('data/meshes/ref.ply')
        if len(target_meshes) > 0:
            target_meshes[0].export('data/target.ply')

    return {
        'zero': zero_mesh,              # for selecting the keypoints (T-pose)
        'template': template_mesh,      # for cutting based on keypoints (modified T-pose (leg spread))
        'ref': ref_mesh,                # the actual reference pose (e.g. A-pose)
        'targets': target_meshes
    }


from loom_core.params import REF_KPTS, REF_KPTS_SKIRTIFIED, process_config
import json
import polyscope as ps


def prepare_ref(design_params, zero_mesh, template_mesh, ref_mesh, garment_part, is_skirtified):
    params_dict = {k: v for key in ['pos', 'length'] for k, v in design_params[garment_part][key].items()}
    ref_kpts = REF_KPTS_SKIRTIFIED if is_skirtified else REF_KPTS
    #full_keypoints_batch, smpl_traversal_pairs, new_mesh = param_to_full_keypoints(t_pose_mesh, REF_KPTS[garment_part], params_dict)
    full_keypoints_batch, smpl_traversal_pairs = param_to_full_keypoints(zero_mesh, ref_kpts[garment_part], params_dict, garment_part, is_skirtified)

    cut_mesh, patches, patch_faces, seamlines_dict_list, symmetric_seamline_flags, valid_patch_idxs = cut_paths(template_mesh, ref_mesh, full_keypoints_batch, smpl_traversal_pairs)
    patch_labels_dict = assign_patch_labels(patches, garment_part, valid_patch_idxs, template_mesh.vertices[SHOULDER_KPT_IDX])

    cut_mesh.export(os.path.join(f'data/meshes/{garment_part}_cut.ply'))

    return cut_mesh, patches, patch_faces, seamlines_dict_list, symmetric_seamline_flags, valid_patch_idxs, patch_labels_dict


def get_pose_adapted(orig_mesh, cut_mesh, patches, patch_faces):
    # NOTE: orig_mesh = meshes_dict['orig]
    target_patches_list = []
    for posed_mesh in meshes_dict['poses']:
        target_pose_mesh = transfer_topology(orig_mesh, cut_mesh, posed_mesh)
        target_patches = extract_target_patches(target_pose_mesh, patches, patch_faces)
        target_patches_list.append(target_patches)
    return target_patches_list


def export_patches(patches, target_patches_list, valid_patch_idxs, garment_part):
    part_patches_dir = f'data/patches/{garment_part}/'
    if os.path.isdir(part_patches_dir):
        shutil.rmtree(part_patches_dir)
    for patch_idx, patch in enumerate(patches):
        if patch_idx in valid_patch_idxs:
            patch_dir = f'{part_patches_dir}/patch_{patch_idx:02d}/'
            os.makedirs(patch_dir)
            for ext in ['ply', 'obj']:  # need OBJ for the flattening optimization
                patch.export(f'{patch_dir}/ref.{ext}')

                for target_idx in range(len(target_patches_list)):
                    target_patches_list[target_idx][patch_idx].export(f'{patch_dir}/target-{target_idx}.{ext}')


def export_seamlines(seamlines_dict_list, symmetric_seamline_flags, garment_part):
    seamline_dir = f'data/seamlines/{garment_part}/'
    if os.path.isdir(seamline_dir):
        shutil.rmtree(seamline_dir)
    os.makedirs(seamline_dir)
    for seamline_idx, seamline_dict in enumerate(seamlines_dict_list):
        for patch_pair in seamline_dict:
            fpath = f'{seamline_dir}/seam-{seamline_idx}_{patch_pair[0]}-{patch_pair[1]}.txt'
            with open(fpath, mode='w') as seam_f:
                seam_f.write('1\n' if symmetric_seamline_flags[seamline_idx] else '0\n')
                seam_f.write(f'{patch_pair[0]}\n{patch_pair[1]}\n')
                for vidx_pair in seamline_dict[patch_pair]:
                    seam_f.write(f'{vidx_pair[0]} {vidx_pair[1]}\n')


def prepare_scales(body_mesh, patch, scale, max_scale, is_skirtified):
    scales_u = np.ones(patch.faces.shape[0]) * scale
    scales_v = np.ones(patch.faces.shape[0])
    
    if max_scale:
        ref_kpts = REF_KPTS_SKIRTIFIED if is_skirtified else REF_KPTS
        top_y = body_mesh.vertices[ref_kpts['lower']['side'][0]][1]
        bottom_y = np.min(patch.vertices[:, 1])
        base_stretch = scale
        max_stretch = max_scale
        
        mean_face_coords = np.mean(patch.vertices[patch.faces], axis=1)
        ref_mask = mean_face_coords[:, 1] < top_y
        scales_u[np.where(ref_mask)] = base_stretch + ((mean_face_coords[ref_mask][:, 1] - top_y) / (bottom_y - top_y)) * (max_stretch - base_stretch)

    return scales_u, scales_v


def export_scales(body_mesh, patches, scale, valid_patch_idxs, garment_part, is_skirtified, max_scale=None):
    part_scales_dir = f'data/scales/{garment_part}/'
    if os.path.isdir(part_scales_dir):
        shutil.rmtree(part_scales_dir)
    for patch_idx, patch in enumerate(patches):
        if patch_idx in valid_patch_idxs:
            patch_scales_dir = os.path.join(part_scales_dir, f'patch_{patch_idx:02d}')
            os.makedirs(patch_scales_dir)

            fpath_u = f'{patch_scales_dir}/scales_u.txt'
            fpath_v = f'{patch_scales_dir}/scales_v.txt'

            scales_u, scales_v = prepare_scales(body_mesh, patch, scale, max_scale, is_skirtified)

            with open(fpath_u, 'w') as f_u:
                for s_u in scales_u:
                    f_u.write(f"{s_u}\n")
            with open(fpath_v, 'w') as f_v:
                for s_v in scales_v:
                    f_v.write(f"{s_v}\n")


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
        os.makedirs(os.path.join(latest_pattern_result_dir, f'patch_{patch_idx:02d}'))
    scales_dir = f'results/scales/{garment_part}/'
    shutil.rmtree(scales_dir)
    os.makedirs(scales_dir)


def run_gif_series():
    import imageio

    if platform == 'darwin':
        PROJECT_DIR = '/Users/kristijanbartol/LOOM/'
        SMPL_DIR = '/Users/kristijanbartol/data/smpl/models/'
    else:
        PROJECT_DIR = '/home/kristijan/LOOM/'
        SMPL_DIR = '/home/kristijan/data/smpl/models/'
    GENDER = 'female'

    with open('config/setup/loom.json') as config_f:
        config = json.load(config_f)

    smpl_model, meshes_dict = prepare_body_meshes(smpl_dir=SMPL_DIR, gender=GENDER)
    experiment_name, design_params, hyperparams, _ = process_config(config)

    # Initialize polyscope
    ps.init()
    gif_images = []

    for param_value in np.arange(0.0, 1.0, 0.2):
        screenshot_dir = f'results/screenshots/upper_side/'
        screenshot_path = os.path.join(screenshot_dir, f'{param_value:.2f}.png')
        if os.path.exists(screenshot_path):
            gif_images.append(imageio.imread(screenshot_path))
        else:
            ps.remove_all_structures()

            ps_body_mesh = ps.register_surface_mesh("body", meshes_dict['orig'].vertices, meshes_dict['orig'].faces, smooth_shade=True)
            ps_body_mesh.set_enabled(False)

            design_params['upper']['pos']['side'] = param_value

            for garment_part in ['upper', 'lower']:
                cut_mesh, patches, patch_faces, seamlines_dict_list, symmetric_seamline_flags, valid_patch_idxs, patch_labels_dict = prepare_ref(design_params, meshes_dict['template'], garment_part)

                export_patches(patches, valid_patch_idxs, garment_part)
                export_seamlines(seamlines_dict_list, symmetric_seamline_flags, garment_part)
                export_scales(patches, valid_patch_idxs, garment_part)
                export_patch_labels(patch_labels_dict, garment_part)
                create_latest_dir(valid_patch_idxs, garment_part)

                garment_mesh = trimesh.util.concatenate([patches[idx] for idx in valid_patch_idxs])
                ps.register_surface_mesh(garment_part, garment_mesh.vertices, garment_mesh.faces, smooth_shade=True)

            ps.reset_camera_to_home_view()
            params = ps.get_view_camera_parameters()
            center = ps.get_view_center()
            pos = np.array(params.get_position())
            new_pos = center + (pos - center) * 0.6   # 0.6x closer
            ps.look_at(tuple(new_pos), tuple(center))

            os.makedirs(screenshot_dir, exist_ok=True)
            ps.screenshot(screenshot_path)

            gif_images.append(imageio.imread(screenshot_path))

            print(f'Processed {screenshot_path}')
            
            #screenshot_default_name = 'screenshot_000000.png'
            
            #shutil.move(os.path.join(PROJECT_DIR, screenshot_default_name), os.path.join(PROJECT_DIR, screenshot_dir, f'{experiment_name}.png'))
            #ps.show()

    #imageio.mimsave(os.path.join(screenshot_dir, 'animation.gif'), gif_images, format='GIF', loop=0)

    #run_loom_optimization()


def run_design(smpl_dir, design_params, body_set, is_dress=False, is_skirt=False):
    is_skirtified = is_dress or is_skirt
    meshes_dict = prepare_body_meshes(smpl_dir=smpl_dir, body_set=body_set, is_skirtified=is_skirtified)
    garment_parts = ['upper'] if is_dress else ['upper', 'lower']

    for garment_part in garment_parts:
        prepared_ref_mesh, ref_patches, patch_faces, seamlines_dict_list, symmetric_seamline_flags, valid_patch_idxs, patch_labels_dict = prepare_ref(
            design_params, meshes_dict['zero'], meshes_dict['template'], meshes_dict['ref'], garment_part, is_skirtified)

        target_patches_list = prepare_targets(meshes_dict['targets'], meshes_dict['ref'], prepared_ref_mesh, ref_patches, patch_faces)

        export_patches(ref_patches, target_patches_list, valid_patch_idxs, garment_part)
        export_seamlines(seamlines_dict_list, symmetric_seamline_flags, garment_part)
        export_scales(meshes_dict['ref'], ref_patches, design_params[garment_part]['scales'], valid_patch_idxs, garment_part, is_skirtified, design_params[garment_part]['max_scale'])
        export_patch_labels(patch_labels_dict, garment_part)
        create_latest_dir(valid_patch_idxs, garment_part)


def prepare_targets(target_meshes, ref_mesh, prepared_ref_mesh, ref_patches, patch_faces):
    target_patches_list = []
    for target_mesh in target_meshes:
        prepared_target_mesh = transfer_topology(ref_mesh, prepared_ref_mesh, target_mesh)
        target_patches_list.append(extract_target_patches(prepared_target_mesh, ref_patches, patch_faces))
    return target_patches_list


if __name__ == '__main__':
    if platform == 'darwin':
        PROJECT_DIR = '/Users/kristijanbartol/LOOM/'
        SMPL_DIR = '/Users/kristijanbartol/data/smpl/models/'
    else:
        PROJECT_DIR = '/home/kristijan/LOOM/'
        SMPL_DIR = '/home/kristijan/data/smpl/models/'

    #GENDER = 'female'

    with open('config/setup/loom.json') as config_f:
        config = json.load(config_f)

    experiment_name, design_params, hyperparams, body_set, scales_dict = process_config(config)
    smpl_model, meshes_dict = prepare_body_meshes(smpl_dir=SMPL_DIR, body_set=body_set)

    for garment_part in ['upper', 'lower']:
        prepared_ref_mesh, ref_patches, patch_faces, seamlines_dict_list, symmetric_seamline_flags, valid_patch_idxs, patch_labels_dict = prepare_ref(
            design_params, meshes_dict['zero'], meshes_dict['template'], meshes_dict['ref'], garment_part)
        
        target_patches_list = prepare_targets(meshes_dict['targets'], meshes_dict['ref'], prepared_ref_mesh, ref_patches, patch_faces)

        export_patches(ref_patches, target_patches_list, valid_patch_idxs, garment_part)
        export_seamlines(seamlines_dict_list, symmetric_seamline_flags, garment_part)
        export_scales(ref_patches, scales_dict[garment_part], valid_patch_idxs, garment_part)
        export_patch_labels(patch_labels_dict, garment_part)
        create_latest_dir(valid_patch_idxs, garment_part)

    run_loom_optimization(hyperparams)
