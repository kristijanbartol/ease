import numpy as np


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