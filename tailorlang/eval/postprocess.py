import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

from tailorlang.const import PATCH_LIST


def add_uv_coordinates(embedded_mesh, uv_coordinates, embedded_mesh_path):
    # Get vertex data from the original mesh
    vertex_data = embedded_mesh['vertex']
    
    # Create a new vertex element with UV coordinates
    vertex_dtype = vertex_data.data.dtype.descr + [('u', 'f4'), ('v', 'f4')]
    
    # Create new vertex array with all properties
    new_vertex_data = np.empty(len(vertex_data.data), dtype=vertex_dtype)
    
    # Copy existing properties
    for prop in vertex_data.data.dtype.names:
        new_vertex_data[prop] = vertex_data.data[prop]
    
    # Add UV coordinates
    new_vertex_data['u'] = uv_coordinates[:, 0]
    new_vertex_data['v'] = uv_coordinates[:, 1]
    
    # Create new vertex element
    vertex_element = PlyElement.describe(new_vertex_data, 'vertex')
    
    # Create new PLY data with updated vertices and original faces
    new_plydata = PlyData([vertex_element, embedded_mesh['face']], text=True)
    
    # Write to new file
    new_plydata.write(embedded_mesh_path)


def postprocess_embedded():
    for patch_label in PATCH_LIST:
        embedded_mesh_path = f'data/embedded/{patch_label}/ref.ply'
        embedded_mesh = PlyData.read(embedded_mesh_path)
        param_2d_mesh = trimesh.load(f'data/param_2d/{patch_label}/optim_final-seams.ply')
        uv_coords = param_2d_mesh.vertices[:, :2]  

        add_uv_coordinates(embedded_mesh, uv_coords, embedded_mesh_path)
