from typing import Dict
import trimesh
import os
import shutil

from loom.sim.utils import (
    process_body_for_simulation,
    process_base_for_simulation,
    process_garment_mesh_for_simulation,
    process_preloaded_body,
    store_garments_for_simulation,
    update_3d_after_simulation,
    update_meshes_after_simulation
)
from loom_core.evaluation.param_mesh_uv import ParamMeshUV
from loom.sim.blender_caller import simulate_pose

from anisotropic_simulations.new_evaluation_frame_based import get_non_skintight_garment


def simulate_garment_set(
        project_dir,
        smpl_dir,
        base_experiment, 
        design_params, 
        body_set, 
        param_mesh_dict: Dict[str, ParamMeshUV],
        optim_dress=False
    ):
    non_skintight_dict_dict = {}
    
    non_skintight_dict_dict['base'] = get_non_skintight_garment(
        project_dir=project_dir,
        experiment_name=base_experiment,
        is_shoulderless=design_params['upper']['flag']['is_shoulderless'],
        optim_dress=optim_dress
    )
    
    garment_parts = ['upper'] if optim_dress else ['upper', 'lower']
    
    # Process garments
    base_param_mesh_dict = {}
    for garment_part in garment_parts:
        garment_data = non_skintight_dict_dict['base'][garment_part]
        garment_mesh = trimesh.Trimesh(
            vertices=garment_data['deformed'], 
            faces=garment_data['embedded'].faces,
            process=False
        )
        param_mesh_dict[garment_part].mesh_3d = garment_mesh
        param_mesh_dict[garment_part].update_duplicate_mesh()
        base_param_mesh_dict[garment_part] = process_base_for_simulation(
            param_mesh=param_mesh_dict[garment_part]
        )   # NOTE: returns param_mesh, useful only to 'base'
    
    # Process body
    ref_body_pose = body_set['poses'][0]
    ref_body_shape = body_set['shapes'][0]
    ref_body_fname = 'ref_x10.ply'
    if len(body_set['poses']) > 1:
        target_body_pose = body_set['poses'][1]
        target_body_shape = body_set['shapes'][1]
        target_fname = 'target-00_x10.ply'
    else:
        target_body_pose = ref_body_pose
        target_body_shape = ref_body_shape
        target_fname = ref_body_fname
        
    body_mesh = process_body_for_simulation(
        smpl_dir=smpl_dir,
        gender=body_set['genders'][0],
        body_pose=target_body_pose,
        body_shape=target_body_shape,
        upper_coef=design_params['upper']['scales'],
        lower_coef=design_params['lower']['scales'],
        optim_dress=optim_dress
    )
    body_path = f'data/body/{target_fname}'
            
    base_upper_path, base_lower_path = store_garments_for_simulation(
        experiment_name=base_experiment,
        body_path=body_path,
        is_refit=False,
        body_mesh=body_mesh,
        upper_param_mesh=base_param_mesh_dict['upper'],
        lower_param_mesh=None if optim_dress else base_param_mesh_dict['lower']
    )
            
    sim_dir = f'results/sim/{base_experiment}/'
    os.makedirs(sim_dir, exist_ok=True)
    shutil.copyfile('data/body/target-00-with-offset.ply', f'results/sim/{base_experiment}/target-00-with-offset.ply')
            
    simulate_pose(
        body_path=body_path,
        shirt_path=base_upper_path,
        pant_path=base_lower_path,
        body_output=os.path.join(sim_dir, 'base_body.ply'),
        shirt_output=os.path.join(sim_dir, 'base_upper.ply'),
        pant_output='' if optim_dress else os.path.join(sim_dir, 'base_lower.ply'),
        is_shoulderless=design_params['upper']['flag']['is_shoulderless'],
        scripts_dir=f'{project_dir}/loom/blender/'
    )   # output_path: results/sim/<base_experiment>/base.ply
    
    qualitative_dir = f'results/qualitative/sim/{base_experiment}/'
    os.makedirs(qualitative_dir, exist_ok=True)
    
    update_meshes_after_simulation(
        sim_dir=sim_dir,
        base_param_mesh_dict=base_param_mesh_dict,
        optim_dress=optim_dress
    )
