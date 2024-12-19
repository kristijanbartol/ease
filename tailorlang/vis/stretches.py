import numpy as np
import trimesh


def _map_stretch_to_color(stretch, min_stretch=0.7, max_stretch=1.3):
    normalized_stretch = (stretch - min_stretch) / (max_stretch - min_stretch)
    intensity = int((1 - normalized_stretch) * 255)
    color = np.array([intensity, intensity, 255 - intensity, 255], dtype=np.uint8)
    return color


def color_code_stretches(verts, faces, stretch_array, min_stretch=0.7, max_stretch=1.3):
    # Ensure the stretch array length matches the number of faces
    assert len(stretch_array) == len(faces), "The length of stretch_array must match the number of faces."

    # Initialize vertex colors
    vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)

    # Count occurrences of each vertex in faces to average the colors
    vertex_counts = np.zeros(verts.shape[0], dtype=np.int32)

    # Apply the color coding
    for face, stretch in zip(faces, stretch_array):
        color = _map_stretch_to_color(stretch, min_stretch, max_stretch)
        for vertex in face:
            vertex_colors[vertex] += color
            vertex_counts[vertex] += 1

    # Average the colors for each vertex
    for i in range(len(vertex_colors)):
        if vertex_counts[i] > 0:
            vertex_colors[i] //= vertex_counts[i]
        else:
            vertex_colors[i] = [128, 128, 128, 255]  # Default gray color if no face uses this vertex

    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)
