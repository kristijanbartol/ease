import os
import shutil
import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

from tailorlang.const import PATCH_LIST


def subdivide_mesh_and_uvs(mesh, uv_coords):
    """
    Subdivide a mesh and its UV coordinates using Loop subdivision.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input 3D mesh to subdivide
    uv_coords : numpy.ndarray
        UV coordinates array of shape (num_vertices, 2)
        
    Returns:
    --------
    new_mesh : trimesh.Trimesh
        The subdivided 3D mesh
    new_uvs : numpy.ndarray
        The subdivided UV coordinates array
    """
    # Verify input dimensions
    if len(uv_coords) != len(mesh.vertices):
        raise ValueError("Number of UV coordinates must match number of vertices")
    
    # Get the vertex indices for each new vertex that will be created
    # This gives us the mapping from old edges to new vertex indices
    new_vertex_indices = mesh.edges_sparse
    
    # Create the subdivided mesh
    new_mesh = mesh.subdivide()
    
    # Initialize the new UV coordinates array
    # The new number of vertices will be:
    # original vertices + one new vertex per edge
    num_new_vertices = len(mesh.vertices) + len(mesh.edges)
    new_uvs = np.zeros((num_new_vertices, 2))
    
    # Copy original UV coordinates
    new_uvs[:len(mesh.vertices)] = uv_coords
    
    # For each edge, compute the UV coordinates of the new vertex
    # as the average of the UV coordinates of the edge endpoints
    for i, edge in enumerate(mesh.edges):
        v1, v2 = edge
        new_vertex_idx = len(mesh.vertices) + i
        new_uvs[new_vertex_idx] = (uv_coords[v1] + uv_coords[v2]) / 2.0
    
    # Now we need to update the positions of the original vertices
    # following the Loop subdivision rules
    beta = 1/8  # Loop's original beta value
    
    # For each original vertex
    for i in range(len(mesh.vertices)):
        # Get neighboring vertices
        neighbors = mesh.vertex_neighbors[i]
        n = len(neighbors)
        
        if n > 0:  # Skip vertices with no neighbors
            # Compute new UV position using Loop's vertex rule
            neighbor_sum = np.sum(uv_coords[neighbors], axis=0)
            new_uvs[i] = (1 - n * beta) * uv_coords[i] + beta * neighbor_sum
    
    return new_mesh, new_uvs


def find_duplicate_vertices(mesh: 'trimesh.Trimesh', 
                          tolerance: float = 1e-4):
    """
    Find vertices that are at the exact same 3D location in a mesh.
    
    Args:
        mesh: trimesh.Trimesh object containing the 3D mesh
        tolerance: float, minimum distance to consider two vertices as duplicates
                  (default: 1e-8 to handle floating point precision issues)
    
    Returns:
        List of tuples, where each tuple contains indices (i, j) of duplicate vertices,
        where i < j to avoid redundant pairs
    """
    vertices = mesh.vertices
    n_vertices = len(vertices)
    duplicate_pairs = []
    
    # Create a dictionary to store vertices rounded to a certain precision
    # This helps handle floating point precision issues
    vertex_dict = {}
    
    for i, vertex in enumerate(vertices):
        # Round the coordinates to handle floating point precision
        vertex_tuple = tuple(np.round(vertex / tolerance) * tolerance)
        
        # If we've seen this vertex before, it's a duplicate
        if vertex_tuple in vertex_dict:
            # Only add the pair if the current index is larger than the stored one
            # This ensures we don't add redundant pairs
            stored_idx = vertex_dict[vertex_tuple]
            duplicate_pairs.append((stored_idx, i))
        else:
            vertex_dict[vertex_tuple] = i
    
    return duplicate_pairs


def trimesh_to_plydata(trimesh_mesh):
    # Extract vertices and faces
    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces

    # Prepare vertices as a structured array
    vertex_array = np.array(
        [(v[0], v[1], v[2]) for v in vertices],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )

    # Prepare faces as a structured array
    face_array = np.array(
        [(list(f),) for f in faces],
        dtype=[('vertex_indices', 'i4', (3,))]
    )

    # Create PlyElement objects for vertices and faces
    vertex_element = PlyElement.describe(vertex_array, 'vertex')
    face_element = PlyElement.describe(face_array, 'face')

    # Create and return a PlyData object
    ply_data = PlyData([vertex_element, face_element], text=True)
    return ply_data


def add_uv_coordinates(mesh_3d, uv_coordinates, output_path):
    if (type(mesh_3d) == trimesh.Trimesh):
        mesh_3d = trimesh_to_plydata(trimesh_mesh=mesh_3d)
        
    # Get vertex data from the original mesh
    vertex_data = mesh_3d['vertex']
    
    # Create a new vertex element with UV coordinates
    vertex_dtype = vertex_data.data.dtype.descr + [('s', 'f4'), ('t', 'f4')]
    
    # Create new vertex array with all properties
    new_vertex_data = np.empty(len(vertex_data.data), dtype=vertex_dtype)
    
    # Copy existing properties
    for prop in vertex_data.data.dtype.names:
        new_vertex_data[prop] = vertex_data.data[prop]
    
    # Add UV coordinates
    new_vertex_data['s'] = uv_coordinates[:, 0]
    new_vertex_data['t'] = uv_coordinates[:, 1]
    
    # Create new vertex element
    vertex_element = PlyElement.describe(new_vertex_data, 'vertex')
    
    # Create new PLY data with updated vertices and original faces
    new_plydata = PlyData([vertex_element, mesh_3d['face']], text=True)
    
    # Write to new file
    new_plydata.write(output_path)


def postprocess(experiment_name):
    # Add uv coords to the embedded patch meshes
    uv_coords_list_dict = {
        'upper': [],
        'lower': []    
    }      # Create uv coords structure to propagate further
    for patch_label in PATCH_LIST:
        embedded_mesh_path = f'data/embedded/{patch_label}/ref.ply'
        embedded_mesh = PlyData.read(embedded_mesh_path)
        param_2d_mesh = trimesh.load(f'results/pattern/{experiment_name}/{patch_label}/optim_final-seams.ply')
        uv_coords = param_2d_mesh.vertices[:, :2]  

        add_uv_coordinates(embedded_mesh, uv_coords, embedded_mesh_path)
        
        if 'lower' in patch_label:
            uv_coords_list_dict['lower'].append(uv_coords)
        else:
            uv_coords_list_dict['upper'].append(uv_coords)
            
    stacked_uv_coords_dict = {}
    stacked_uv_coords_dict['upper'] = np.vstack(uv_coords_list_dict['upper'])
    stacked_uv_coords_dict['lower'] = np.vstack(uv_coords_list_dict['lower'])
    
    # Copy latest patches to the current experiment folder (results/pattern/latest/ -> results/pattern/<experiment>/)
    pattern_2d_dir = 'results/pattern/'
    latest_dir = os.path.join(pattern_2d_dir, 'latest/')
    experiment_dir = os.path.join(pattern_2d_dir, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    for patch_label in os.listdir(latest_dir):
        shutil.copytree(os.path.join(latest_dir, patch_label), os.path.join(experiment_dir, patch_label), dirs_exist_ok=True)
        
    return stacked_uv_coords_dict
