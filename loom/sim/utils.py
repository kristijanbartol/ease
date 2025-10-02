import os
import numpy as np
import trimesh
from smplx import SMPL

from loom import const
from loom.const import apply_angle_offset


def process_preloaded_body(body_mesh):
    # TODO: implement an automatic joint angle offsetting method
    body_mesh.vertices *= 10.
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0] 
    ))
    
    return body_mesh


def process_body_for_simulation(smpl_dir, gender, body_pose, body_shape, upper_coef, lower_coef, optim_dress):
    pose_label = str.replace(body_pose, '-', '_')
    smpl_model = SMPL(model_path=os.path.join(smpl_dir, f'SMPL_{gender.upper()}.pkl'), gender=gender)
    pose_params = getattr(const, pose_label)()
    shape_params = getattr(const, body_shape)()

    apply_angle_offset(
        pose_params=pose_params,
        pose_label=pose_label,
        upper_coef=upper_coef,
        lower_coef=lower_coef,
        optim_dress=optim_dress
    )
    
    body_verts = smpl_model(body_pose=pose_params, betas=shape_params).vertices[0].cpu().detach().numpy()
    body_mesh = trimesh.Trimesh(vertices=body_verts, faces=smpl_model.faces)
    
    body_mesh.export('data/body/target-00-with-offset.ply')
        
    body_mesh.vertices *= 10.
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0] 
    ))
    
    return body_mesh


def process_garment_mesh_for_simulation(garment_mesh):
    garment_mesh.vertices *= 10.
    garment_mesh.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0] 
    ))
    garment_mesh.subdivide()
    return garment_mesh


def process_base_for_simulation(param_mesh):
    param_mesh.mesh_3d_with_duplicates.vertices *= 10.
    param_mesh.mesh_3d_with_duplicates.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0] 
    ))
    param_mesh.subdivide()
    return param_mesh


def postprocess_3d_after_simulation(sim_mesh, output_path):
    sim_mesh.apply_transform(trimesh.transformations.rotation_matrix(
        angle=-np.pi/2,
        direction=[1, 0, 0] 
    ))
    sim_mesh /= 10.
    sim_mesh.export(output_path)


def postprocess_base_after_simulation(
        param_mesh, 
        sim_mesh: trimesh.Trimesh, 
        output_path: str
    ) -> None:
    param_mesh.mesh_3d_subdivided.vertices = sim_mesh.vertices
    param_mesh.update_duplicate_mesh_subdivided()
    param_mesh.mesh_3d_with_duplicates_subdivided.apply_transform(trimesh.transformations.rotation_matrix(
        angle=-np.pi/2,
        direction=[1, 0, 0] 
    ))
    param_mesh.mesh_3d_with_duplicates_subdivided.vertices /= 10.
    param_mesh.export(output_path)


def process_refit_for_simulation(garment_mesh: trimesh.Trimesh):
    garment_mesh.vertices *= 10.
    garment_mesh.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0] 
    ))
    garment_mesh.subdivide()
    return garment_mesh


def store_garments_for_simulation(
        experiment_name,    # base experiment name
        body_path,     # the path to the body mesh on top of which garments will be draped
        is_refit,         # the pose to which to refit, 'base' in case of not processing the refit pose
        body_mesh,          # target-01.ply -> transformed
        upper_param_mesh,         # upper trimesh mesh (merged)
        lower_param_mesh = None,          # lower trimesh mesh (merged)
    ):
    non_skintight_garment_dir = f'results/non-skintight/{experiment_name}'
    os.makedirs(non_skintight_garment_dir, exist_ok=True)
    
    body_mesh.export(body_path)
    
    basename = 'refit' if is_refit else 'base'
    
    upper_path = os.path.join(non_skintight_garment_dir, f'{basename}_upper.ply')
    lower_path = os.path.join(non_skintight_garment_dir, f'{basename}_lower.ply')
    upper_path_with_uv = os.path.join(non_skintight_garment_dir, f'{basename}_upper_uv.ply')
    lower_path_with_uv = os.path.join(non_skintight_garment_dir, f'{basename}_lower_uv.ply')
    
    if is_refit:
        upper_param_mesh.export(upper_path)
        if lower_param_mesh is not None:
            lower_param_mesh.export(lower_path)
    else:
        upper_param_mesh.mesh_3d_subdivided.export(upper_path)
        upper_param_mesh.export(upper_path_with_uv)     # mesh with UV and duplicates for mid-result and rendering
        if lower_param_mesh is not None:
            lower_param_mesh.mesh_3d_subdivided.export(lower_path)
            lower_param_mesh.export(lower_path_with_uv)     # mesh with UV and duplicates for mid-result and rendering
    
    return upper_path, lower_path


def update_3d_after_simulation(
        sim_dir, 
        garment_mesh_dict
    ):
    # Update simulation results to include uv coordinates as well + apply inverse transformations
    upper_mesh = trimesh.load(os.path.join(sim_dir, 'base_upper.ply'))
    lower_mesh = trimesh.load(os.path.join(sim_dir, 'base_lower.ply'))
    
    postprocess_3d_after_simulation(
        sim_mesh=upper_mesh, 
        output_path=os.path.join(sim_dir, 'upper.ply')
    )
    postprocess_3d_after_simulation(
        sim_mesh=lower_mesh, 
        output_path=os.path.join(sim_dir, 'lower.ply')
    )


def update_meshes_after_simulation(
        sim_dir, 
        base_param_mesh_dict,
        optim_dress
    ):
    # Update simulation results to include uv coordinates as well + apply inverse transformations
    upper_mesh = trimesh.load(os.path.join(sim_dir, 'base_upper.ply'))
    if not optim_dress:
        lower_mesh = trimesh.load(os.path.join(sim_dir, 'base_lower.ply'))
    
    postprocess_base_after_simulation(
        param_mesh=base_param_mesh_dict['upper'], 
        sim_mesh=upper_mesh, 
        output_path=os.path.join(sim_dir, 'base_upper_uv.ply')
    )
    if not optim_dress:
        postprocess_base_after_simulation(
            param_mesh=base_param_mesh_dict['lower'], 
            sim_mesh=lower_mesh, 
            output_path=os.path.join(sim_dir, 'base_lower_uv.ply')
        )
