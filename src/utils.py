import trimesh
import os
import numpy as np

from src.const import COLOR_MAP


def update_color_indices(garment_vertex_indices, color_dict):
    # Create a mapping from old SMPL indices to new garment mesh indices
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(garment_vertex_indices)}
    
    # Update the color dictionary with the new indices
    new_color_dict = {}
    for color, indices in color_dict.items():
        # Update the indices using the mapping, filter out indices not found in the mapping
        new_indices = [index_mapping.get(index) for index in indices if index in index_mapping]
        # Remove any None values that were not in the mapping
        new_indices = [index for index in new_indices if index is not None]
        new_color_dict[color] = new_indices
    
    return new_color_dict


def write_ply_file(filename, verts, faces, vertex_colors):
    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for vert in verts:
            color = vertex_colors.get(tuple(vert), 'gray')  # Default color is gray
            r, g, b = COLOR_MAP.get(color, (128, 128, 128))  # Convert color name to RGB
            f.write(f"{vert[0]} {vert[1]} {vert[2]} {r} {g} {b}\n")
        
        # Write face list
        for face in faces:
            f.write(f"3 {' '.join(str(v) for v in face)}\n")


def write_obj_file(filename, verts, faces):
    with open(filename, 'w') as f:
        # Write vertices
        for vert in verts:
            f.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        
        # Write faces
        # OBJ files use 1-based indexing, so we need to add 1 to each vertex index
        for face in faces:
            f.write(f"f {' '.join(str(v + 1) for v in face)}\n")


def export_to_ply(args, verts, faces, vertex_indices_by_color, path):
    ply_filename = f"{path}.ply"
    if args.standard_export:
        trimesh.Trimesh(vertices=verts, faces=faces).export(ply_filename)
    else:
        # Map each vertex to its color
        vertex_colors = {tuple(verts[i]): color for color, indices in vertex_indices_by_color.items() for i in indices}
        
        # Make sure faces are referring to the correct vertex indices
        used_vertex_indices = set(idx for indices in vertex_indices_by_color.values() for idx in indices)
        verts = verts[list(used_vertex_indices)]
        
        # Update the face indices
        index_mapping = {old_index: new_index for new_index, old_index in enumerate(sorted(used_vertex_indices))}
        updated_faces = [[index_mapping[idx] for idx in face] for face in faces if set(face).issubset(used_vertex_indices)]

        write_ply_file(ply_filename, verts, updated_faces, vertex_colors)


def export_to_obj(args, verts, faces, vertex_indices_by_color, path):
    obj_filename = f"{path}.obj"
    if args.standard_export:
        trimesh.Trimesh(vertices=verts, faces=faces).export(obj_filename)
    else:
        # Make sure faces are referring to the correct vertex indices
        used_vertex_indices = set(idx for indices in vertex_indices_by_color.values() for idx in indices)
        verts = verts[list(used_vertex_indices)]
        
        # Update the face indices
        index_mapping = {old_index: new_index for new_index, old_index in enumerate(sorted(used_vertex_indices))}
        updated_faces = [[index_mapping[idx] for idx in face] for face in faces if set(face).issubset(used_vertex_indices)]

        write_obj_file(obj_filename, verts, updated_faces)


def export(args, verts, faces, path, format='ply', vertex_indices_by_color=None):
    if format == 'ply':
        export_to_ply(args, verts, faces, vertex_indices_by_color, path)
    elif format == 'obj':
        export_to_obj(args, verts, faces, vertex_indices_by_color, path)
    else:
        export_to_ply(args, verts, faces, vertex_indices_by_color, path)
        export_to_obj(args, verts, faces, vertex_indices_by_color, path)


def color_code_stretches(verts, faces, stretch_array, min_stretch=0.7, max_stretch=1.3):
    # Ensure the stretch array length matches the number of faces
    assert len(stretch_array) == len(faces), "The length of stretch_array must match the number of faces."
    
    def map_stretch_to_color(stretch):
        normalized_stretch = (stretch - min_stretch) / (max_stretch - min_stretch)
        intensity = int((1 - normalized_stretch) * 255)
        color = np.array([intensity, intensity, 255 - intensity, 255], dtype=np.uint8)
        return color

    # Initialize vertex colors
    vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)

    # Count occurrences of each vertex in faces to average the colors
    vertex_counts = np.zeros(verts.shape[0], dtype=np.int32)

    # Apply the color coding
    for face, stretch in zip(faces, stretch_array):
        color = map_stretch_to_color(stretch)
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


def extract_local_stretches(verts, faces, design_dict, garment_part, side=None) -> np.ndarray:

    if design_dict[garment_part]['type'] == 'uniform':
        stretches_u = np.ones(faces.shape[0]) * design_dict[garment_part]['base_stretch_u']
        stretches_v = np.ones(faces.shape[0]) * design_dict[garment_part]['base_stretch_v']

    elif design_dict[garment_part]['type'] == 'linear':
        stretches_u, stretches_v = [], []
        mean_face_coords = np.mean(verts[faces], axis=1)
        if garment_part == 'sleeve':
            min_x = np.min(verts[:, 0])
            max_x = np.max(verts[:, 0])
            stretches_u = np.ones(faces.shape[0]) * design_dict['sleeve']['base_stretch_u']
            base_stretch, max_stretch = design_dict['sleeve']['base_stretch_v'], design_dict['sleeve']['max_stretch']
            if side == 'left':
                stretches_v = base_stretch + ((mean_face_coords[:, 0] - min_x) / (max_x - min_x)) * (max_stretch - base_stretch)
            else:
                stretches_v = base_stretch + ((mean_face_coords[:, 0] - max_x) / (min_x - max_x)) * (max_stretch - base_stretch)
        else:
            min_y = np.min(verts[:, 1])
            max_y = np.max(verts[:, 1])
            base_stretch, max_stretch = design_dict[garment_part]['base_stretch_u'], design_dict[garment_part]['max_stretch']
            stretches_u = base_stretch + ((mean_face_coords[:, 0] - max_y) / (min_y - max_y)) * (max_stretch - base_stretch)
            stretches_v = np.ones(faces.shape[0]) * design_dict[garment_part]['base_stretch_v']

    elif design_dict[garment_part]['type'] == 'linear_from':
        stretches_u, stretches_v = [], []
        mean_face_coords = np.mean(verts[faces], axis=1)
        if garment_part == 'sleeve':
            ref_x = design_dict[garment_part]['ref_x']
            min_x = np.min(verts[:, 0])
            max_x = np.max(verts[:, 0])
            stretches_u = np.ones(faces.shape[0]) * design_dict['sleeve']['base_stretch_u']
            base_stretch, max_stretch = design_dict['sleeve']['base_stretch_v'], design_dict['sleeve']['max_stretch']
            if side == 'left':
                ref_mask = mean_face_coords[:, 0] > ref_x
                stretches_v[np.where(ref_mask)] = base_stretch + ((mean_face_coords[ref_mask][:, 0] - ref_x) / (max_x - ref_x)) * (max_stretch - base_stretch)
                stretches_v[np.where(~ref_mask)] = design_dict['sleeve']['base_stretch_v']
            else:
                ref_mask = mean_face_coords[:, 0] < -ref_x
                stretches_v[np.where(ref_mask)] = base_stretch + ((mean_face_coords[ref_mask][:, 0] - max_x) / (ref_x - max_x)) * (max_stretch - base_stretch)
                stretches_v[np.where(~ref_mask)] = design_dict['sleeve']['base_stretch_v']
        else:
            ref_y = design_dict[garment_part]['ref_y']
            min_y = np.min(verts[:, 1])
            max_y = np.max(verts[:, 1])
            base_stretch, max_stretch = design_dict[garment_part]['base_stretch_u'], design_dict[garment_part]['max_stretch']
            ref_mask = mean_face_coords[:, 1] < ref_y
            stretches_u[np.where(ref_mask)] = base_stretch + ((mean_face_coords[ref_mask][:, 0] - ref_y) / (min_y - ref_y)) * (max_stretch - base_stretch)
            stretches_u[np.where(~ref_mask)] = design_dict[garment_part]['base_stretch_u']
            stretches_v = np.ones(faces.shape[0]) * design_dict[garment_part]['base_stretch_v']

    else:
        print('WARNING: Wrong design stretch type, returning uniform (1.0).')
    
    return stretches_u, stretches_v


def read_ref_shape(shape_idx):
    return np.load(f'data/shapes/params{shape_idx:03d}.npy')


def save_seamline_pairs_file(fpath, seamline_pair_dict):
    id1, id2 = list(seamline_pair_dict.keys())
    vertex_idxs1 = seamline_pair_dict[id1]
    vertex_idxs2 = seamline_pair_dict[id2]

    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    with open(fpath, 'w') as f:
        f.write(f"{id1}\n")
        f.write(f"{id2}\n")
        
        for idx1, idx2 in zip(vertex_idxs1, vertex_idxs2):
            f.write(f"{idx1} {idx2}\n")
