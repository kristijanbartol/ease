import trimesh
import os

from loom.sim.utils import (
    process_garment_mesh_for_simulation,
    process_preloaded_body,
    update_3d_after_simulation
)
from loom.sim.blender_caller import simulate_pose

from anisotropic_simulations.evaluation_frame_based import get_non_skintight_garment


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