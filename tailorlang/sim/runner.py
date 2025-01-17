import trimesh
import os

from tailorlang.utils import modify_experiment_name_for_refit
from tailorlang.sim.utils import (
    process_body_for_simulation,
    process_garment_for_simulation,
    store_garments_for_simulation
)
from tailorlang.sim.blender_caller import simulate_pose
from tailorlang.eval.postprocess import add_uv_coordinates

from anisotropic_simulations.evaluation_frame_based import get_non_skintight_garment


def simulate_garment_set(config, base_experiment, design_params, body_set, stacked_uv_coords_dict):
    non_skintight_dict_dict = {}
    refit_pose = None if config.refit_pose == '' else config.refit_pose
    
    non_skintight_dict_dict['base'] = get_non_skintight_garment(
        project_dir=config.project_dir,
        experiment_name=base_experiment
    )
    
    if refit_pose:
        refit_experiment = modify_experiment_name_for_refit(base_experiment, refit_pose)
        non_skintight_dict_dict['refit'] = get_non_skintight_garment(
            project_dir=config.project_dir,
            experiment_name=refit_experiment
        )
        
    # Preprocess meshes for the simulation
    preprocessed_garments_dict = {}
    #merged_garments_dict = {}
    #duplicate_pairs_dict = {}
    
    # Process garments
    for keyword in non_skintight_dict_dict:
        preprocessed_garments_dict[keyword] = {}
        #merged_garments_dict[keyword] = {}
        for garment_part in non_skintight_dict_dict[keyword]:
            garment_data = non_skintight_dict_dict[keyword][garment_part]
            garment_mesh = trimesh.Trimesh(
                vertices=garment_data['deformed'], 
                faces=garment_data['embedded'].faces,
                process=False
            )
            preprocessed_garments_dict[keyword][garment_part] = process_garment_for_simulation(
                garment_mesh=garment_mesh
            )       # The UV coords subdivision update only works for the base experiment, which is sufficient for now
    
    # Process body
    if body_set.num_targets == 0:
        body_pose = body_set.ref['pose']
        body_shape = body_set.ref['shape']
        body_fname = 'ref_x10.ply'
    else:
        body_pose = body_set.target['poses'][0]
        body_shape = body_set.target['shapes'][0]
        body_fname = 'target-00_x10.ply'
        
    body_mesh = process_body_for_simulation(
        smpl_dir=config.smpl_dir,
        gender=body_set.ref_gender,
        body_pose=body_pose,
        body_shape=body_shape,
        upper_coef=design_params.shirt_looseness,
        lower_coef=design_params.pant_looseness
    )
    body_path = f'data/body/{body_fname}'
            
    base_upper_path, base_lower_path = store_garments_for_simulation(
        experiment_name=base_experiment,
        body_path=body_path,
        refit_pose='base',
        body_mesh=body_mesh,
        upper_mesh=preprocessed_garments_dict['base']['upper'],
        lower_mesh=preprocessed_garments_dict['base']['lower']
    )
    
    if refit_pose:
        refit_upper_path, refit_lower_path = store_garments_for_simulation(
            experiment_name=base_experiment,    # store in the base experiment since you use base pose
            body_path=body_path,
            refit_pose=refit_pose,
            body_mesh=body_mesh,
            upper_mesh=preprocessed_garments_dict['refit']['upper'],
            lower_mesh=preprocessed_garments_dict['refit']['lower']
        )
            
    simulate_pose(
        body_path=body_path,
        shirt_path=base_upper_path,
        pant_path=base_lower_path,
        body_output='results/sim/base_body.ply',
        shirt_output='results/sim/base_shirt.ply',
        pant_output='results/sim/base_pant.ply',
        blender_path='/Applications/Blender.app/Contents/MacOS/Blender',
        scripts_dir=f'{config.project_dir}/tailorlang/blender/'
    )   # output_path: results/sim/<base_experiment>/base.ply
    
    if refit_pose:
        simulate_pose(
            body_path=body_path,
            shirt_path=refit_upper_path,
            pant_path=refit_lower_path,
            body_output='results/sim/refit_body.ply',
            shirt_output='results/sim/refit_shirt.ply',
            pant_output='results/sim/refit_pant.ply',
            blender_path='/Applications/Blender.app/Contents/MacOS/Blender',
            scripts_dir=f'{config.project_dir}/tailorlang/blender/'
        )   # output_path: results/sim/<base_experiment>/refit.ply
