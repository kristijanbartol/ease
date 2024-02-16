from const import COLOR_MAP


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


def export_to_ply(verts, faces, vertex_indices_by_color, filename_prefix):
    # Map each vertex to its color
    vertex_colors = {tuple(verts[i]): color for color, indices in vertex_indices_by_color.items() for i in indices}
    
    # Make sure faces are referring to the correct vertex indices
    used_vertex_indices = set(idx for indices in vertex_indices_by_color.values() for idx in indices)
    verts = verts[list(used_vertex_indices)]
    
    # Update the face indices
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(sorted(used_vertex_indices))}
    updated_faces = [[index_mapping[idx] for idx in face] for face in faces if set(face).issubset(used_vertex_indices)]

    # Write to PLY
    ply_filename = f"{filename_prefix}.ply"
    write_ply_file(ply_filename, verts, updated_faces, vertex_colors)


def export_to_obj(verts, faces, vertex_indices_by_color, filename_prefix):
    # Make sure faces are referring to the correct vertex indices
    used_vertex_indices = set(idx for indices in vertex_indices_by_color.values() for idx in indices)
    verts = verts[list(used_vertex_indices)]
    
    # Update the face indices
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(sorted(used_vertex_indices))}
    updated_faces = [[index_mapping[idx] for idx in face] for face in faces if set(face).issubset(used_vertex_indices)]

    # Write to PLY
    ply_filename = f"{filename_prefix}.obj"
    write_obj_file(ply_filename, verts, updated_faces)


def export(verts, faces, vertex_indices_by_color, filename_prefix, format='ply'):
    if format == 'ply':
        export_to_ply(verts, faces, vertex_indices_by_color, filename_prefix)
    else:
        export_to_obj(verts, faces, vertex_indices_by_color, filename_prefix)
