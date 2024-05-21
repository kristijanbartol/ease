import trimesh
import numpy as np
from scipy.spatial import KDTree


def modify_mesh_based_on_plane_cut(mesh, point, label):
    # Convert the input point to a numpy array if it isn't already
    point = np.array(point)
    
    # Define the normal of the vertical plane
    if label == 'left':
        normal = np.array([1, 0, 0])  # Plane normal pointing to the left
    elif label == 'right':
        normal = np.array([-1, 0, 0])  # Plane normal pointing to the right
    else:
        raise ValueError("Label must be 'left' or 'right'")

    # Create the plane equation
    plane = np.append(normal, -np.dot(normal, point))

    # Find the intersection of the mesh with the plane
    slice_lines = mesh.section(plane_origin=point, plane_normal=normal)
    
    if slice_lines is None:
        print("No intersections found with the mesh.")
        return mesh
    
    # Get the vertices of the intersection lines
    slice_points = np.vstack(slice_lines.vertices)

    # Identify the vertices to be modified
    vertices_to_modify = []

    for face in mesh.faces:
        for vertex_index in face:
            vertex = mesh.vertices[vertex_index]
            # Check if the vertex is "outside" the plane
            if (label == 'left' and vertex[0] < point[0]) or (label == 'right' and vertex[0] > point[0]):
                vertices_to_modify.append(vertex_index)
    
    # Remove duplicates
    vertices_to_modify = list(set(vertices_to_modify))

    # Modify the vertices of the intersected faces
    for vertex_index in vertices_to_modify:
        vertex = mesh.vertices[vertex_index]
        # Find the closest point on the slice to this vertex
        distances = np.linalg.norm(slice_points - vertex, axis=1)
        closest_point = slice_points[np.argmin(distances)]
        # Move the vertex to the intersection point
        mesh.vertices[vertex_index] = closest_point
    
    return mesh



def cut_and_project_mesh(mesh, point, side):
    # Create a vertical plane defined by the given point
    normal = np.array([1, 0, 0])  # Normal for a vertical plane (YZ plane)
    d = -np.dot(normal, point)
    
    # Function to determine if a vertex is "outside"
    def is_outside(vertex):
        if side == 'left':
            return vertex[0] < point[0]
        elif side == 'right':
            return vertex[0] > point[0]
        else:
            raise ValueError("Invalid side. Use 'left' or 'right'.")

    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    
    # Iterate over each face to find intersections with the plane
    for face in faces:
        # Get the vertices of the face
        v0, v1, v2 = vertices[face]
        
        # Check if the face is intersected by the plane
        d0 = np.dot(v0, normal) + d
        d1 = np.dot(v1, normal) + d
        d2 = np.dot(v2, normal) + d
        
        if (d0 < 0 and d1 > 0 and d2 > 0) or (d0 > 0 and d1 < 0 and d2 < 0) or \
           (d1 < 0 and d0 > 0 and d2 > 0) or (d1 > 0 and d0 < 0 and d2 < 0) or \
           (d2 < 0 and d0 > 0 and d1 > 0) or (d2 > 0 and d0 < 0 and d1 < 0):
            # This face intersects the plane, project "outside" vertices
            if is_outside(v0):
                vertices[face[0]][0] = point[0]
            if is_outside(v1):
                vertices[face[1]][0] = point[0]
            if is_outside(v2):
                vertices[face[2]][0] = point[0]

    # Return the full, modified mesh
    modified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return modified_mesh


# Example usage
if __name__ == "__main__":
    # Load an example SMPL body mesh
    smpl_mesh = trimesh.load('body_mesh.ply')

    # Define a point for the vertical plane
    plane_point = [0.28, 0.0, 0.0]

    # Label for the side ('left' or 'right')
    label = 'left'

    # Modify the mesh based on the plane cut
    #modified_mesh = modify_mesh_based_on_plane_cut(smpl_mesh, plane_point, label)
    modified_mesh = cut_and_project_mesh(smpl_mesh, plane_point, label)

    # Save the modified mesh
    modified_mesh.export('modified_smpl_mesh.ply')
