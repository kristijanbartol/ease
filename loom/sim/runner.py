from typing import Dict
import trimesh
import os
import shutil

from loom.utils import modify_experiment_name_for_refit
from loom.sim.utils import (
    process_body_for_simulation,
    process_base_for_simulation,
    process_refit_for_simulation,
    process_garment_mesh_for_simulation,
    process_preloaded_body,
    store_garments_for_simulation,
    update_3d_after_simulation,
    update_meshes_after_simulation,
    ParamMeshUV
)
from loom.sim.blender_caller import simulate_pose

from anisotropic_simulations.evaluation_frame_based import get_non_skintight_garment


def simulate_garment_set(
        config, 
        base_experiment, 
        design_params, 
        body_set, 
        param_mesh_dict: Dict[str, ParamMeshUV],
        optim_dress=False
    ):
    non_skintight_dict_dict = {}
    refit_pose = None if config.refit_pose == '' else config.refit_pose
    
    non_skintight_dict_dict['base'] = get_non_skintight_garment(
        project_dir=config.project_dir,
        experiment_name=base_experiment,
        optim_dress=config.optim_dress
    )
    
    if refit_pose:
        refit_experiment = modify_experiment_name_for_refit(base_experiment, refit_pose)
        non_skintight_dict_dict['refit'] = get_non_skintight_garment(
            project_dir=config.project_dir,
            experiment_name=refit_experiment
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
    ref_body_pose = body_set.ref['pose']
    ref_body_shape = body_set.ref['shape']
    ref_body_fname = 'ref_x10.ply'
    if body_set.num_targets > 0:
        target_body_pose = body_set.target['poses'][0]
        target_body_shape = body_set.target['shapes'][0]
        target_fname = 'target-00_x10.ply'
    else:
        target_body_pose = ref_body_pose
        target_body_shape = ref_body_shape
        target_fname = ref_body_fname
        
    body_mesh = process_body_for_simulation(
        smpl_dir=config.smpl_dir,
        gender=body_set.ref_gender,
        body_pose=target_body_pose,
        body_shape=target_body_shape,
        upper_coef=design_params.shirt_looseness,
        lower_coef=design_params.pant_looseness,
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
    
    if refit_pose:
        refit_preprocessed_garment_dict = {}
        for garment_part in garment_parts:
            garment_data = non_skintight_dict_dict['refit'][garment_part]
            garment_mesh = trimesh.Trimesh(
                vertices=garment_data['deformed'], 
                faces=garment_data['embedded'].faces,
                process=False
            )
            refit_preprocessed_garment_dict[garment_part] = process_refit_for_simulation(
                garment_mesh=garment_mesh
            )
        
        refit_upper_path, refit_lower_path = store_garments_for_simulation(
            experiment_name=base_experiment,    # store in the base experiment since you use base pose
            body_path=body_path,
            is_refit=True,
            body_mesh=body_mesh,
            upper_param_mesh=refit_preprocessed_garment_dict['upper'],
            lower_param_mesh=None if optim_dress else refit_preprocessed_garment_dict['lower']
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
        blender_path='/Applications/Blender.app/Contents/MacOS/Blender',
        scripts_dir=f'{config.project_dir}/tailorlang/blender/'
    )   # output_path: results/sim/<base_experiment>/base.ply
    
    qualitative_dir = f'results/qualitative/sim/{base_experiment}/'
    os.makedirs(qualitative_dir, exist_ok=True)
    
    update_meshes_after_simulation(
        sim_dir=sim_dir,
        base_param_mesh_dict=base_param_mesh_dict,
        optim_dress=optim_dress
    )
    
    if refit_pose:
        simulate_pose(
            body_path=body_path,
            shirt_path=refit_upper_path,
            pant_path=refit_lower_path,
            body_output=os.path.join(sim_dir, 'refit_body.ply'),
            shirt_output=os.path.join(sim_dir, 'refit_upper.ply'),
            pant_output=None if optim_dress else os.path.join(sim_dir, 'refit_lower.ply'),
            blender_path='/Applications/Blender.app/Contents/MacOS/Blender',
            scripts_dir=f'{config.project_dir}/tailorlang/blender/'
        )   # output_path: results/sim/<base_experiment>/refit.ply


def simulate_3d(config, experiment_name):
    non_skintight_mesh_dict = get_non_skintight_garment(
        project_dir=config.project_dir,
        experiment_name=experiment_name
    )

    target_body_pose = body_set.ref['pose']
    target_body_shape = body_set.ref['shape']
    target_fname = 'ref_x10.ply'

    body_mesh = process_preloaded_body(body_mesh)
    body_path = f'data/body/{target_fname}'
    body_mesh.export(body_path)

    garment_mesh_dict = {}
    garment_path_dict = {}
    for garment_part in ['upper', 'lower']:
        garment_data = non_skintight_mesh_dict[garment_part]
        garment_mesh_dict[garment_part] = trimesh.Trimesh(
            vertices=garment_data['deformed'], 
            faces=garment_data['embedded'].faces,
            process=False
        )
        garment_mesh_dict[garment_part] = process_garment_mesh_for_simulation(
            garment_mesh_dict[garment_part]
        )
        garment_path_dict[garment_part] = f'results/non-skintight/{experiment_name}/{garment_part}.ply'
        garment_mesh_dict[garment_part].export(garment_path_dict[garment_part])

    sim_dir = f'results/sim/{experiment_name}/'
    os.makedirs(sim_dir, exist_ok=True)

    simulate_pose(
        body_path=body_path,
        shirt_path=garment_path_dict['upper'],
        pant_path=garment_path_dict['lower'],
        body_output=os.path.join(sim_dir, 'base_body.ply'),
        shirt_output=os.path.join(sim_dir, 'base_upper.ply'),
        pant_output=os.path.join(sim_dir, 'base_lower.ply'),
        blender_path='/Applications/Blender.app/Contents/MacOS/Blender',
        scripts_dir=f'{config.project_dir}/tailorlang/blender/'
    )   # output_path: results/sim/<base_experiment>/base.ply

    update_3d_after_simulation(sim_dir, garment_mesh_dict)
