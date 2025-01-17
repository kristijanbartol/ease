import os
import numpy as np
import trimesh
from smplx import SMPL

from tailorlang import const
from tailorlang.const import apply_angle_offset
from tailorlang.eval.postprocess import (
    add_uv_coordinates,
    find_duplicate_vertices,
    subdivide_mesh_and_uvs
)


def process_body_for_simulation(smpl_dir, gender, body_pose, body_shape, upper_coef, lower_coef):
    pose_label = str.replace(body_pose, '-', '_')
    smpl_model = SMPL(model_path=os.path.join(smpl_dir, f'SMPL_{gender.upper()}.pkl'), gender=gender)
    pose_params = getattr(const, pose_label)()
    shape_params = getattr(const, body_shape)()

    apply_angle_offset(
        pose_params=pose_params,
        pose_label=pose_label,
        upper_coef=upper_coef,
        lower_coef=lower_coef
    )
    
    body_verts = smpl_model(body_pose=pose_params, betas=shape_params).vertices[0].cpu().detach().numpy()
    body_mesh = trimesh.Trimesh(vertices=body_verts, faces=smpl_model.faces)
        
    body_mesh.vertices *= 10.
    body_mesh.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0] 
    ))
    
    return body_mesh


def process_garment_for_simulation(garment_mesh):
    garment_mesh.vertices *= 10.
    garment_mesh.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0] 
    ))
    #duplicate_pairs = find_duplicate_vertices(garment_mesh)
    #merged_garment_mesh = trimesh.Trimesh(
    #    vertices=garment_mesh.vertices,
    #    faces=garment_mesh.faces,
    #    process=True
    #)
    garment_mesh = garment_mesh.subdivide()
    #garment_mesh, stacked_uv_coords = subdivide_mesh_and_uvs(
    #    mesh=garment_mesh,
    #    uv_coords=stacked_uv_coords
    #)
    return garment_mesh


def store_garments_for_simulation(
        experiment_name,    # base experiment name
        body_path,     # the path to the body mesh on top of which garments will be draped
        refit_pose,         # the pose to which to refit, 'base' in case of not processing the refit pose
        body_mesh,          # target-01.ply -> transformed
        upper_mesh,         # upper trimesh mesh (merged)
        lower_mesh,          # lower trimesh mesh (merged)
    ):
    non_skintight_garment_dir = f'results/non-skintight/{experiment_name}'
    os.makedirs(non_skintight_garment_dir, exist_ok=True)
    
    upper_path = os.path.join(non_skintight_garment_dir, f'{refit_pose}_upper.ply')
    lower_path = os.path.join(non_skintight_garment_dir, f'{refit_pose}_lower.ply')
    
    body_mesh.export(body_path)
    upper_mesh.export(upper_path)
    lower_mesh.export(lower_path)
    
    return upper_path, lower_path
