import trimesh
import numpy as np


def modify_mesh_with_plane(vertices, faces, point):
    # Calculate the normal of the vertical plane
    plane_normal = np.array([1, 0, 0])  # Assuming the plane is vertical and normal is along the x-axis
    plane_point = np.array(point)
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

    modified_mesh = trimesh.Trimesh(vertices=new_vertices, faces=faces)
    return modified_mesh


# Example usage
if __name__ == "__main__":
    # Load an example SMPL body mesh
    #smpl_mesh = trimesh.load('body_mesh.ply')
    smpl_mesh = trimesh.load('results/target_meshes/set2/body-00.ply')

    # Define a point for the vertical plane
    plane_point = [0.3, 0.0, 0.0]

    # Label for the side ('left' or 'right')
    label = 'left'

    # Modify the mesh based on the plane cut
    modified_mesh = modify_mesh_with_plane(smpl_mesh.vertices, smpl_mesh.faces, plane_point)

    # Save the modified mesh
    modified_mesh.export('modified_smpl_mesh.ply')
