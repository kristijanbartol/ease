from typing import List
import trimesh
import os
import numpy as np

from tailorlang.const import (
    COLOR_MAP,
    DISPLACEMENTS,
    PATCH_LIST,
    SEGMENT_NAMES
)


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
            stretches_u = base_stretch + ((mean_face_coords[:, 1] - max_y) / (min_y - max_y)) * (max_stretch - base_stretch)
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


def save_darts_files(dirpath, darts_dict_for_segment):
    for dart_name in darts_dict_for_segment:
        vertex_pairs = darts_dict_for_segment[dart_name]
        with open(os.path.join(dirpath, f'{dart_name}.txt'), 'w') as f:
            f.write(f"{vertex_pairs[0]}\n")
            
            for idx1, idx2 in vertex_pairs[1:]:
                f.write(f"{idx1} {idx2}\n")


def export_body_mesh(args, verts, faces, set_element_idx, mesh_set_dir, latest_set_dir, gender):
    body_colors = {'gray': list(range(len(verts)))}
    export(args, verts, faces, os.path.join(mesh_set_dir, 'body', f'body-{set_element_idx:02d}'), args.file_format, body_colors)
    export(args, verts, faces, os.path.join(latest_set_dir, 'body', f'body-{set_element_idx:02d}'), args.file_format, body_colors)
    export(args, verts, faces, os.path.join('data/simulated/houdini/latest', gender), args.file_format, body_colors)


def export_color_coded_designs(args, garment, design_dict, garment_parts, posed_verts, mesh_set_dir, latest_set_dir):
    upper_indices = (garment_parts['upper_front'] + garment_parts['upper_back'] + 
                     garment_parts['sleeve_front_right'] + garment_parts['sleeve_back_right'] + 
                     garment_parts['sleeve_front_left'] + garment_parts['sleeve_back_left'])
    lower_indices = sum([garment_parts[f'lower_{part}'] for part in ['front_right', 'front_left', 'back_right', 'back_left']], [])

    upper_garment_verts, upper_garment_faces = garment.extract_garment_mesh(posed_verts, garment.mesh.faces, upper_indices, offset=DISPLACEMENTS['skintight'])
    lower_garment_verts, lower_garment_faces = garment.extract_garment_mesh(posed_verts, garment.mesh.faces, lower_indices, offset=DISPLACEMENTS['skintight'])

    upper_garment_stretch_array_u, _ = extract_local_stretches(upper_garment_verts, upper_garment_faces, design_dict['stretches'], 'upper')
    lower_garment_stretch_array_u, _ = extract_local_stretches(lower_garment_verts, lower_garment_faces, design_dict['stretches'], 'lower')

    upper_garment_mesh = color_code_stretches(upper_garment_verts, upper_garment_faces, upper_garment_stretch_array_u)
    lower_garment_mesh = color_code_stretches(lower_garment_verts, lower_garment_faces, lower_garment_stretch_array_u)

    upper_garment_mesh.export(f'{mesh_set_dir}/upper_garment_init.ply')
    lower_garment_mesh.export(f'{mesh_set_dir}/lower_garment_init.ply')
    upper_garment_mesh.export(f'{latest_set_dir}/upper_garment_init.ply')
    lower_garment_mesh.export(f'{latest_set_dir}/lower_garment_init.ply')


def create_directories(mesh_set_dir, latest_set_dir):
    garment_part_names = [
        'upper_front', 'upper_back',
        'sleeve_front_right', 'sleeve_back_right',
        'sleeve_front_left', 'sleeve_back_left',
        'lower_front_right', 'lower_front_left',
        'lower_back_right', 'lower_back_left'
    ]

    for part_name in garment_part_names:
        os.makedirs(os.path.join(mesh_set_dir, part_name), exist_ok=True)
        os.makedirs(os.path.join(latest_set_dir, part_name), exist_ok=True)

    # Create directories for body meshes and color-coded designs
    os.makedirs(os.path.join(mesh_set_dir, 'body'), exist_ok=True)
    os.makedirs(os.path.join(latest_set_dir, 'body'), exist_ok=True)
    os.makedirs(os.path.join('data/simulated/houdini/latest'), exist_ok=True)


def export_stretch_arrays(design_dict, verts, faces, part_name, mesh_set_dir, latest_set_dir):
    stretch_array_u, stretch_array_v = extract_local_stretches(
        verts=verts,
        faces=faces,
        design_dict=design_dict['stretches'],
        garment_part=part_name.split('_')[0]
    )

    np.savetxt(f'{mesh_set_dir}/{part_name}/stretches_u.txt', stretch_array_u)
    np.savetxt(f'{mesh_set_dir}/{part_name}/stretches_v.txt', stretch_array_v)
    np.savetxt(f'{latest_set_dir}/{part_name}/stretches_u.txt', stretch_array_u)
    np.savetxt(f'{latest_set_dir}/{part_name}/stretches_v.txt', stretch_array_v)


def export_edge_lengths(garment):
    
    def process_garment_part(segment_names: List, garment):
        mesh_dict = {}
        for segment_name in segment_names:
            subdir_path = os.path.join(result_dir, segment_name)
            result_mesh_path = os.path.join(subdir_path, 'optim_final-seams.ply')
            optimized_mesh = trimesh.load(result_mesh_path)
            mesh_dict[segment_name] = optimized_mesh
            
        return garment.measure_optimized_edge_lengths(mesh_dict)
    
    def save_edge_map(root_dir, part_name, edge_lengths_map):
        file_path = os.path.join(root_dir, f'{part_name}_optim_edge_lengths.txt')
        with open(file_path, 'w') as f:
            for (int1, int2), value in edge_lengths_map.items():
                f.write(f"{int1} {int2} {value:.6f}\n")
    
    segment_names_dict = {}
    segment_names_dict['upper'] = SEGMENT_NAMES['default']['upper']
    segment_names_dict['lower'] = SEGMENT_NAMES['default']['lower']
    
    result_dir = 'data/embedded/latest/skintight/'
    for part_name in ['upper', 'lower']:
        edge_lengths_map = process_garment_part(segment_names_dict[part_name], garment)
        save_edge_map(result_dir, part_name, edge_lengths_map)


def store_preselected(template_dir_path, patch_idxs_dict, modified_verts):
    for patch_label in patch_idxs_dict:
        np.save(os.path.join(template_dir_path, f'{patch_label}_idxs.npy'), np.array(patch_idxs_dict[patch_label], dtype=np.int32))    
    np.save(os.path.join(template_dir_path, 'modified_verts.npy'), modified_verts)


def load_preselected(template_dir_path):
    patch_idxs_dict = {}
    for patch_label in PATCH_LIST:
        patch_idxs_dict[patch_label] = np.load(os.path.join(template_dir_path, f'{patch_label}.npy'))
    modified_verts = np.load(os.path.join(template_dir_path, 'modified_verts.npy'))

    return patch_idxs_dict, modified_verts
