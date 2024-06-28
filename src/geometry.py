import numpy as np
import scipy
import trimesh
import torch
import random

from src.const import (
    PLANE_ORIENT_DICT,
    SEAM_IDX_DICT,
    THRESH_TO_SEAMS_DICT,
    THRESH_TO_SEGMENT_DICT
)
from src.seams import extract_parameterized_seams


def compute_vertex_normals(verts, faces):
    # Calculate the normals for each face
    face_normals = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                            verts[faces[:, 2]] - verts[faces[:, 0]])
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, np.newaxis]

    # Initialize vertex normals as zero
    vertex_normals = np.zeros_like(verts)

    # Add face normals to each vertex normal
    for i, face in enumerate(faces):
        vertex_normals[face] += face_normals[i]

    # Normalize the vertex normals
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]

    return vertex_normals


def apply_offset_to_verts(verts, faces, offset):
    # Compute the vertex normals
    vertex_normals = compute_vertex_normals(verts, faces)

    # Apply the offset to each vertex along its normal
    offset_verts = verts + vertex_normals * offset

    return offset_verts


def bezier_curve(t, control_points):
    """ Compute a point on a Bezier curve with given control points and parameter t """
    n = len(control_points) - 1
    return sum(
        scipy.special.comb(n, i) * (1 - t)**(n - i) * t**i * control_points[i]
        for i in range(n + 1)
    )


def project_points_to_nearest_vertices(points, mesh):
    """
    Project each point in 'points' to the nearest vertex on 'mesh'.
    :param points: A list or a numpy array of points (Nx3).
    :param mesh: A trimesh.Trimesh object representing the mesh.
    :return: A numpy array of points projected onto the nearest vertices of the mesh.
    """
    # KDTree for efficient nearest neighbor search
    #kdtree = trimesh.kdtree.KDTree(mesh.vertices)
    kdtree = scipy.spatial.KDTree(mesh.vertices)

    # Find the nearest vertex for each point
    _, vertex_indices = kdtree.query(points)

    # Project points onto the nearest vertices
    projected_points = mesh.vertices[vertex_indices].copy()
    return projected_points


def project_boundaries_using_faces_deprecated(mesh, points):
    # Project points onto the nearest faces
    _, _, triangle_ids = trimesh.proximity.closest_point(mesh, points)

    # Select all the vertices within the selected triangles as boundaries
    boundary_vertex_idxs = []
    for triangle_id in triangle_ids:
        boundary_vertex_idxs.extend(mesh.faces[triangle_id])
    return boundary_vertex_idxs


def project_boundaries(mesh, points):
    """Project Bezier curves to the mesh surface.
    
    Extract set of boundary faces (provided as keys in the returning dictionary).
    Extract list of boundary points, used to traverse along the boundary when 
    selecting the starting points for warp/weft directions.
    Create a face-to-boundary-points dictionary for quick access to the corresponding
    boundary points, given the face.
    """
    boundary_points, _, boundary_face_ids = trimesh.proximity.closest_point(mesh, points)
    face_id_to_points_dict = {}
    boundary_vertex_ids = []
    for idx, face_id in enumerate(boundary_face_ids):
        # The tuple stores the current projected boundary point, as well as the left
        # and right neighboring ones.
        face_id_to_points_dict[face_id] = (
            boundary_points[idx-1],
            boundary_points[idx],
            boundary_points[(idx+1)%boundary_face_ids.shape[0]]
        )
        boundary_vertex_ids.extend(mesh.faces[face_id])
    return face_id_to_points_dict, boundary_points, boundary_vertex_ids


def extract_boundaries(
        args,
        orig_verts,
        sub_verts,
        body_part_keypoints_dict,
        num_points
    ):
    # TODO: Process all the body parts.
    BODY_PART = 'upper_front'
    t_values = np.linspace(0, 1, num_points)
    body_part_curve_points = np.empty((0, 3))
    bottom_left_point, bottom_right_point = None, None
    # Iterate over boundaries of the body part
    for boundary_part in body_part_keypoints_dict:
        control_point_idx_list = body_part_keypoints_dict[boundary_part]
        control_points = [orig_verts[idx] for idx in control_point_idx_list]

        # Select Bezier control points based on the length parameter
        if boundary_part == 'left_side':
            _, bottom_left_point = extract_parameterized_seams(
                verts=sub_verts, 
                garment_length=args.shirt_length, 
                seam_vertex_indices=SEAM_IDX_DICT[BODY_PART]['left_armpit']
            )
            control_points.append(bottom_left_point)
        if boundary_part == 'right_side':
            _, bottom_right_point = extract_parameterized_seams(
                verts=sub_verts, 
                garment_length=args.shirt_length, 
                seam_vertex_indices=SEAM_IDX_DICT[BODY_PART]['right_armpit']
            )
            control_points.append(bottom_right_point)
        if boundary_part == 'bottom':
            mid_point = (bottom_left_point + bottom_right_point) / 2
            mid_point[2] += 0.1
            control_points.extend([bottom_left_point, mid_point, bottom_right_point])

        # Obtain Bezier curve points and concatenate to the list of unprojected boundaries
        body_part_curve_points = np.vstack([
            body_part_curve_points,
            np.array([bezier_curve(t, control_points) for t in t_values])
        ])
    return body_part_curve_points


def find_init_face_deprecated(mesh, start_point):
    # Offset the provided starting point in the Z direction to project it to the mesh surface.
    start_point[2] += 0.1
    # Project from the offset point and find a corresponding triangle ID.
    triangle_id = trimesh.proximity.closest_point(mesh, np.expand_dims(start_point, 0))[2][0]
    # Select any vertex from the triangle specified by the ID (for example, vertex 0).
    return triangle_id


def find_init_vertex_idx(mesh, start_point):
    # Offset the provided starting point in the Z direction to project it to the mesh surface.
    start_point[2] += 0.1
    # Project from the offset point and find a corresponding triangle ID.
    triangle_id = trimesh.proximity.closest_point(mesh, np.expand_dims(start_point, 0))[2][0]
    # Select any vertex from the triangle specified by the ID (for example, vertex 0).
    return mesh.faces[triangle_id][random.randint(0, 2)]


def modify_mesh_with_plane_cut(vertices, faces, cutting_point, plane_orientation, sleeve_side=None):
    if plane_orientation == 'vertical':
        plane_side = -1 if sleeve_side == 'right' else 1
        plane_normal = np.array([plane_side, 0, 0])
        plane_point = np.array([cutting_point, 0, 0])
    elif plane_orientation == 'horizontal':
        plane_normal = np.array([0, -1, 0])
        plane_point = np.array([0, cutting_point, 0])
    
    plane_d = -np.dot(plane_normal, plane_point)

    def point_plane_distance(vertex):
        return np.dot(plane_normal, vertex) + plane_d

    def is_intersecting(v1, v2):
        d1 = point_plane_distance(v1)
        d2 = point_plane_distance(v2)
        return d1 * d2 < 0

    def intersection_point(v1, v2):
        d1 = point_plane_distance(v1)
        d2 = point_plane_distance(v2)
        t = d1 / (d1 - d2)
        return v1 + t * (v2 - v1)

    def move_to_nearest_edge(vertex, polygon_edges):
        min_dist = float('inf')
        nearest_point = vertex
        for edge in polygon_edges:
            p1, p2 = edge
            edge_vec = p2 - p1
            t = np.dot(vertex - p1, edge_vec) / np.dot(edge_vec, edge_vec)
            t = np.clip(t, 0, 1)
            projection = p1 + t * edge_vec
            dist = np.linalg.norm(vertex - projection)
            if dist < min_dist:
                min_dist = dist
                nearest_point = projection
        return nearest_point

    intersected_triangles = []
    intersection_points = []

    for face in faces:
        v1, v2, v3 = vertices[face]
        intersections = []
        if is_intersecting(v1, v2):
            intersections.append(intersection_point(v1, v2))
        if is_intersecting(v2, v3):
            intersections.append(intersection_point(v2, v3))
        if is_intersecting(v3, v1):
            intersections.append(intersection_point(v3, v1))
        
        if len(intersections) == 2:
            intersection_points.extend(intersections)
            intersected_triangles.append(face)

    if not intersection_points:
        return vertices, faces

    intersection_points = np.array(intersection_points)
    polygon_edges = [(intersection_points[i], intersection_points[i + 1]) for i in range(0, len(intersection_points) - 1, 2)]

    new_vertices = vertices.copy()
    for face in intersected_triangles:
        for i in range(3):
            v_idx = face[i]
            vertex = vertices[v_idx]
            if point_plane_distance(vertex) > 0:
                new_vertices[v_idx] = move_to_nearest_edge(vertex, polygon_edges)

    return new_vertices


def cut_seamlines(design_dict, verts, faces):
    threshold_dict = {}
    for threshold_label in THRESH_TO_SEGMENT_DICT:
        body_part_label = threshold_label.split('_')[0]
        seams_label = THRESH_TO_SEAMS_DICT[threshold_label]
        segment_label = THRESH_TO_SEGMENT_DICT[threshold_label]

        _, threshold = extract_parameterized_seams(
            verts=verts,
            garment_length=design_dict['dims'][body_part_label],
            seam_idx_dict=SEAM_IDX_DICT[segment_label][seams_label]
        )
        threshold_dict[threshold_label] = threshold
        verts = modify_mesh_with_plane_cut(
            vertices=verts,
            faces=faces,
            cutting_point=threshold,
            plane_orientation=PLANE_ORIENT_DICT[threshold_label]
        )
    return verts, threshold_dict
