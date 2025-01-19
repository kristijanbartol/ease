import os
import shutil
import trimesh
from plyfile import PlyData

from tailorlang.const import PATCH_LIST
from tailorlang.sim.utils import (
    add_uv_coordinates,
    ParamMeshUV
)


def postprocess(experiment_name):
    # Copy latest patches to the current experiment folder (results/pattern/latest/ -> results/pattern/<experiment>/)
    pattern_2d_dir = 'results/pattern/'
    latest_dir = os.path.join(pattern_2d_dir, 'latest/')
    experiment_dir = os.path.join(pattern_2d_dir, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    for patch_label in os.listdir(latest_dir):
        shutil.copytree(os.path.join(latest_dir, patch_label), os.path.join(experiment_dir, patch_label), dirs_exist_ok=True)
    
    embedded_mesh_list_dict = {
        'upper': [],
        'lower': []    
    } 
    param_2d_mesh_list_dict = {
        'upper': [],
        'lower': []    
    } 
    for patch_label in PATCH_LIST:
        embedded_mesh_path = f'data/embedded/{patch_label}/ref.ply'
        embedded_mesh = trimesh.load(embedded_mesh_path)
        embedded_mesh_plydata = PlyData.read(embedded_mesh_path)
        param_2d_mesh = trimesh.load(f'results/pattern/{experiment_name}/{patch_label}/optim_final-seams.ply')
        uv_coords = param_2d_mesh.vertices[:, :2]  

        add_uv_coordinates(embedded_mesh_plydata, uv_coords, embedded_mesh_path)
        
        # Add to the UV coords list
        if 'lower' in patch_label:
            param_2d_mesh_list_dict['lower'].append(param_2d_mesh)
        else:
            param_2d_mesh_list_dict['upper'].append(param_2d_mesh)
        
        # Add to the embedded mesh list
        if 'lower' in patch_label:
            embedded_mesh_list_dict['lower'].append(embedded_mesh)
        else:
            embedded_mesh_list_dict['upper'].append(embedded_mesh)
            
    param_mesh_dict = {
        'upper': ParamMeshUV(
            mesh_3d_list=embedded_mesh_list_dict['upper'],
            mesh_2d_list=param_2d_mesh_list_dict['upper'],
            garment_part='upper'
        ),
        'lower': ParamMeshUV(
            mesh_3d_list=embedded_mesh_list_dict['lower'],
            mesh_2d_list=param_2d_mesh_list_dict['lower'],
            garment_part='lower'
        )
    }
        
    return param_mesh_dict
