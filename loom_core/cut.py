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

from utils import insert_midline_point, horizontal_plane_cut


'''
// A map that specified the polyline category for the polylinesBatch<*> vectors.
std::map<std::string, std::vector<size_t>> categoryToPolylinesMap = {
    { "seamlines", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} },
    { "boundaries", {12, 13, 14, 15} }
};

std::vector<bool> isLoop = {false,false,false,false,false,false,false,false,false,false,false,false,false,false,true};

// The upper and lower vector specifying keypoints for extracting the geodesic polylines.
std::vector<std::vector<size_t>> polylinesBatchUpper = {
    // "seamlines"
        {4767, 4962, 4310},         // right armpit         0
        {1285, 1323, 823},          // left armpit          1
        {6469, 4305},               // right shoulder       2
        {3010, 817},                // left shoulder        3
        {6469, 5085, 5480},         // right sleeve up      4
        {4767, 5099, 5568},         // right sleeve down    5
        {3010, 1915, 2019},         // left sleeve up       6
        {1285, 1917, 2060},         // left sleeve down     7
        {4767, 4162, 6469},         // right front arm      8
        {1285, 674, 3010},          // left front arm       9
        {6469, 5345, 4200, 4767},   // right back arm       10
        {3011, 1891, 713, 1285},    // left back arm        11
        
    // "boundaries",
        {4305, 4058, 570, 817},     // front neck           12
        {817, 1219, 4301, 4305},    // back neck            13
        {4310, 823},                // lower boundary (front): extracted using right and left armpit keypoints                          14
        {823, 3484, 4310}           // lower boundary (back): need to pick another (middle) vertex to force traverseing the back side   15
};

std::vector<std::vector<size_t>> polylinesBatchLowerMap = {
    {6378, 4460, 4943, 6869},           // right outer pant
    {2919, 979, 1117, 3469},            // left outer pant
    {1208, 4439, 4591, 6833},           // right inner pant
    {1208, 1028, 1122, 3433},           // left inner pant
    {3507, 3510, 1208},                 // front inner pant
    {1784, 3119, 1476, 3170, 1208}      // right sleeve down
};
'''


# NOTE: Using three coordinates for some polylines to direct the path to the correct side.
#       The third coordinates are changing the extract geodesic path.
polylines_upper_batch = [
    [4767, 4310],               # right armpit                  0
    [1285, 823],                # left armpit                   1
    [6469, 4305],               # right shoulder                2
    [3010, 817],                # left shoulder                 3
    [6469, 5480],               # right sleeve up               4
    [3010, 2019],               # left sleeve up                5
    [4767, 5568],               # right sleeve down             6
    [1285, 2060],               # left sleeve down              7
    [4767, 6469],               # right front arm               8
    [1285, 3011],               # left front arm                9
    [6469, 5345, 4767],         # right back arm                10
    [3011, 713, 1285],          # left back arm                 11

    [4305, 3060, 817],          # front neck                    12
    [817, 1219, 4305],          # back neck                     13
    [4310, 823],                # lower boundary (front)        14
    [823, 3484, 4310],          # lower boundary (back)         15
    [5480, 5568],               # right sleeve boundary         16
    [2019, 2060],               # left sleeve boundary          17
    [5480, 5695, 5568],         # right sleeve boundary (up)    18
    [2019, 2234, 2060]          # left sleeve boundary (up)     19
]


keypoints_batch1 = [
    [4767, 4310],
    [6469, 4305],
    [6469, 5480],
    [4767, 5568],
    [4767, 6469],
    [6469, 5345, 4767],

    [4305, 3060],
    [817, 1219],
    [4310],
    [5480, 5695, 5568]
]


class TraversalType(Enum):
    GEODESIC = auto()
    SHORTEST_PATH = auto()
    CIRCULAR = auto()
    PLANE_CUT = auto()


# NOTE: for symmetric designs, all keypoints are given on one side, otherwise, they are considered assymetric
ref_keypoints_dict1 = {
    'mid': [3168, 3500],
    'neck': [4294, 5310],
    'shoulder': [5282, 5335],
    'armpit': [5326, 4891],

    'sleeve': None,
    'bottom': None
}

params1 = {
    'mid': 0.3,         # [0., 1.]
    'neck': 0.5,        # [0., 1.]
    'shoulder': 0.5,    # [0., 1.]
    'armpit': 0.5,      # [0., 1.]

    'sleeve': 0.2,      # [cm]
    'bottom': 0.25        # [cm]
}

traversal_types1 = {
    'mid': TraversalType.GEODESIC,
    'neck': TraversalType.SHORTEST_PATH,
    'shoulder': TraversalType.CIRCULAR,
    'armpit': TraversalType.GEODESIC,

    'sleeve': TraversalType.GEODESIC,
    'bottom': TraversalType.PLANE_CUT
}

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
    return path[int(interp * float((len(path))))]


def _extract_parametric_keypoint(mesh, starting_idx, dir_vector, offset, length: float):
    query_p = mesh.vertices[starting_idx] + offset + dir_vector * length
    dists = np.linalg.norm(mesh.vertices - query_p, axis=1)
    closest_vertex_index = np.argmin(dists)
    return closest_vertex_index


def _param_to_core_keypoints(mesh, ref_keypoints_dict, params_dict):
    core_idx_dict = {}
    for k in ref_keypoints_dict:
        if ref_keypoints_dict[k]:
            core_idx_dict[k] = _extract_keypoint_along_path(mesh, ref_keypoints_dict[k][0], ref_keypoints_dict[k][1], params_dict[k])
    
    # sleeves - optional
    if params_dict['sleeve'] and core_idx_dict['shoulder']:
        dir_vector = np.array([-1, 0, 0])
        sleeve_length = params_dict['sleeve']
        core_idx_dict['sleeve_up'] = _extract_parametric_keypoint(mesh, core_idx_dict['shoulder'], dir_vector, np.array([0., 0.00, 0.]), sleeve_length)
        core_idx_dict['sleeve_down'] = _extract_parametric_keypoint(mesh, core_idx_dict['armpit'], dir_vector, np.zeros(3,), sleeve_length)

    # bottom - mandatory
    core_idx_dict['bottom'] = _extract_parametric_keypoint(mesh, core_idx_dict['armpit'], np.array([0, -1, 0]), np.zeros(3,), params_dict['bottom'])

    return core_idx_dict


def _core_to_side_keypoints(mesh, core_idxs_dict):
    side_keypoints_batch = []

    # close sleeve boundary
    if 'sleeve_up' in core_idxs_dict:
        kpt_idx1, kpt_idx2 = core_idxs_dict['sleeve_up'], core_idxs_dict['sleeve_down']
        front_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, 0.01)
        back_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, -0.1)
        side_keypoints_batch.append([kpt_idx1, front_idx, kpt_idx2])
        side_keypoints_batch.append([kpt_idx1, back_idx,  kpt_idx2])

        side_keypoints_batch.append([core_idxs_dict['shoulder'], core_idxs_dict['sleeve_up']])
        side_keypoints_batch.append([core_idxs_dict['armpit'], core_idxs_dict['sleeve_down']])

    # mid-neck
    side_keypoints_batch.append([core_idxs_dict['neck'], core_idxs_dict['mid']])

    # back neck
    V, F, mid_back_idx = insert_midline_point(mesh.vertices, mesh.faces, core_idxs_dict['neck'], front=False)
    side_keypoints_batch.append([core_idxs_dict['neck'], mid_back_idx])

    # neck-shoulder
    side_keypoints_batch.append([core_idxs_dict['neck'], core_idxs_dict['shoulder']])

    # shoulder loop
    kpt_idx1, kpt_idx2 = core_idxs_dict['shoulder'], core_idxs_dict['armpit']
    front_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, 0.05)
    back_idx = _extract_side_idx(mesh, kpt_idx1, kpt_idx2, -0.1)
    side_keypoints_batch.append([kpt_idx1, front_idx, kpt_idx2])
    side_keypoints_batch.append([kpt_idx1, back_idx, kpt_idx2])

    # armpit-bottom
    side_keypoints_batch.append([core_idxs_dict['armpit'], core_idxs_dict['bottom']])

    # bottom
    V, F, mid_front_idx = insert_midline_point(V, F, core_idxs_dict['bottom'], front=True)
    V, F, mid_back_idx  = insert_midline_point(V, F, core_idxs_dict['bottom'], front=False)
    side_keypoints_batch.append([core_idxs_dict['bottom'], mid_front_idx])
    side_keypoints_batch.append([core_idxs_dict['bottom'], mid_back_idx])

    #V, F, _ = horizontal_plane_cut(V, F, V[core_idxs_dict['bottom']][1])

    return side_keypoints_batch, V, F


def param_to_full_keypoints(mesh, ref_keypoints_dict, params_dict):
    core_idxs_dict = _param_to_core_keypoints(mesh, ref_keypoints_dict, params_dict)
    side_keypoints_batch, newV, newF = _core_to_side_keypoints(mesh, core_idxs_dict)

    new_mesh = trimesh.Trimesh(vertices=newV, faces=newF)
    full_keypoints_batch = _side_to_full_keypoints(new_mesh, side_keypoints_batch)
    return full_keypoints_batch, new_mesh


def flood_fill_vertex_patches_with_multilabels(V, F, boundary_verts, polylines):
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    boundary_set = set(boundary_verts)

    # Build adjacency excluding boundaries
    adjacency = {i: set() for i in range(len(V))}
    for face in F:
        for i in range(3):
            vi = face[i]
            vj = face[(i + 1) % 3]
            if vi in boundary_set or vj in boundary_set:
                continue
            adjacency[vi].add(vj)
            adjacency[vj].add(vi)

    labels = -1 * np.ones(len(V), dtype=int)
    current_label = 0

    for v_start in range(len(V)):
        if v_start in boundary_set or labels[v_start] != -1:
            continue
        queue = deque([v_start])
        labels[v_start] = current_label

        while queue:
            v = queue.popleft()
            for nbr in adjacency[v]:
                if labels[nbr] == -1:
                    labels[nbr] = current_label
                    queue.append(nbr)
        current_label += 1

    # Assign final labels ("ordinary" + boundary).
    final_labels = [[] for _ in range(len(V))]
    for i, lbl in enumerate(labels):
        if lbl != -1:
            final_labels[i].append(lbl)

    # Assign multi-labels to boundary vertices.
    for bv in boundary_verts:
        neighbor_labels = set()
        for face_id in mesh.vertex_faces[bv]:
            for vi in F[face_id]:
                if vi not in boundary_set:
                    neighbor_labels.update(final_labels[vi])
        final_labels[bv] = sorted(neighbor_labels)

    # TODO: Process boundary vertices that appear in more than one seamline (T- and X-junctions).
    bv_counter = {v: 0 for v in boundary_verts}
    for polyline_verts in polylines:
        for bv in polyline_verts:
            bv_counter[bv] += 1

    for bv in bv_counter:
        if bv_counter[bv] > 1:
            neighbor_labels = set()
            for face_id in mesh.vertex_faces[bv]:
                for vi in F[face_id]:
                    neighbor_labels.update(final_labels[vi])
            final_labels[bv] = sorted(neighbor_labels)

    return final_labels


def extract_and_save_patch_meshes(V, F, vertex_labels, prefix='patch'):
    os.makedirs('patches/', exist_ok=True)

    # Face → label mapping: assign face to a patch if **all 3 vertices share that label**
    patch_faces = defaultdict(list)

    for face in F:
        v0, v1, v2 = face
        # Find common labels across the 3 vertices
        common_labels = set(vertex_labels[v0]) & set(vertex_labels[v1]) & set(vertex_labels[v2])
        for lbl in common_labels:
            patch_faces[lbl].append(face)

    # Write out each patch
    for patch_id, face_list in patch_faces.items():
        face_array = np.array(face_list)
        # Find unique vertex indices
        unique_verts, inverse_indices = np.unique(face_array.flatten(), return_inverse=True)

        excluded_patch = False
        for excl_v in exclude_patch_vidxs:
            if excl_v in unique_verts:
                excluded_patch = True

        #if not excluded_patch:
        V_patch = V[unique_verts]
        F_patch = inverse_indices.reshape((-1, 3))
        mesh_patch = trimesh.Trimesh(vertices=V_patch, faces=F_patch, process=False)
        mesh_patch.export(os.path.join(f"patches/{prefix}_{patch_id}.ply"))


def cut_paths(mesh, keypoints_batch):
    V = mesh.vertices
    F = mesh.faces

    path_solver = pp3d.ExtendedEdgeFlipGeodesicSolver(V, F)
    keypoint_coordinates = []
    for pair in keypoints_batch:
        keypoint_coordinates.append([V[pair[0]], V[pair[-1]]])
    newV, newF, geodesic_indices = path_solver.apply_cuts(keypoints_batch, keypoint_coordinates)

    flat_indices = [x for xs in geodesic_indices for x in xs]
    v_labels = flood_fill_vertex_patches_with_multilabels(newV, newF, flat_indices, geodesic_indices)
    extract_and_save_patch_meshes(newV, newF, v_labels)

    trimesh.PointCloud(vertices=newV[flat_indices]).export('cuts.ply')
    for i, geodesic_idxs in enumerate(geodesic_indices):
        trimesh.PointCloud(vertices=newV[geodesic_idxs]).export(f'cuts_{i}.ply')

    return trimesh.Trimesh(vertices=newV, faces=newF)


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
    orig_verts = smpl_model(
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

    #subdiv_verts, subdiv_faces = trimesh.remesh.subdivide_loop(orig_verts, smpl_faces, iterations=1)
    #subdiv_mesh = trimesh.Trimesh(subdiv_verts, subdiv_faces)
    full_keypoints_batch, new_mesh = param_to_full_keypoints(orig_mesh, ref_keypoints_dict1, params1)

    #full_keypoints_batch = _side_to_full_keypoints(orig_mesh, keypoints_batch1)
    cut_mesh = cut_paths(new_mesh, full_keypoints_batch)

    transfer_shape = transfer_topology(orig_mesh, cut_mesh, shaped_mesh)
    transfer_pose = transfer_topology(orig_mesh, cut_mesh, posed_mesh)

    orig_mesh.export('orig_mesh.ply')
    new_mesh.export('new_mesh.ply')
    #subdiv_mesh.export('subdiv_mesh.ply')
    cut_mesh.export('cut_mesh.ply')
    transfer_shape.export('transfer_shape.ply')
    transfer_pose.export('transfer_pose.ply')
