import os
import torch
from smplx import SMPL

from src.geometry import modify_mesh_with_plane_cut


def initialize_smpl_models(smpl_dir):
    smpl_models = {}
    smpl_models['male'] = SMPL(model_path=os.path.join(smpl_dir, 'SMPL_MALE.pkl'), gender='male')
    smpl_models['female'] = SMPL(model_path=os.path.join(smpl_dir, 'SMPL_FEMALE.pkl'), gender='female')
    return smpl_models


def modify_body_mesh(garment, seams_info):
    verts = garment.mesh.vertices
    faces = garment.mesh.faces

    # Upper cut
    modified_verts = modify_mesh_with_plane_cut(
        vertices=verts,
        faces=faces,
        cutting_point=seams_info['y_upper_threshold'],
        plane_orientation='horizontal'
    )

    # Update y-coordinates after upper cut
    y_lower_threshold_low = min(v[1] for v in modified_verts if v[1] > seams_info['y_upper_threshold'])
    y_lower_threshold_up = seams_info['y_lower_threshold_up']

    # Lower cuts
    modified_verts = modify_mesh_with_plane_cut(
        vertices=modified_verts,
        faces=faces,
        cutting_point=y_lower_threshold_low,
        plane_orientation='horizontal'
    )
    if y_lower_threshold_up is not None:
        modified_verts = modify_mesh_with_plane_cut(
            vertices=modified_verts,
            faces=faces,
            cutting_point=y_lower_threshold_up,
            plane_orientation='horizontal'
        )

    # Sleeve cuts
    for side in ['left', 'right']:
        sleeve_threshold = seams_info[f'sleeve_front_{side}']['threshold']
        modified_verts = modify_mesh_with_plane_cut(
            vertices=modified_verts,
            faces=faces,
            cutting_point=sleeve_threshold,
            plane_orientation='vertical',
            sleeve_side=side
        )

    # Update seams_info with new thresholds
    seams_info['y_lower_threshold_low'] = y_lower_threshold_low

    return modified_verts, seams_info


def initialize_modified_smpl_models(smpl_dir, modified_verts):
    modified_models = {}
    for gender in ['male', 'female']:
        modified_models[gender] = SMPL(
            model_path=os.path.join(smpl_dir, f'SMPL_{gender.upper()}.pkl'),
            gender=gender,
            v_template=torch.from_numpy(modified_verts).float()
        )
    return modified_models
