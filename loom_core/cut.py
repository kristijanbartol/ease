import potpourri3d as pp3d
from smplx import SMPL
import os
from sys import platform
import torch


def cut_single_path(V, F):
    path_solver = pp3d.EdgeFlipGeodesicSolver(V, F) # shares precomputation for repeated solves
    path_pts = path_solver.find_geodesic_path_poly(v_list=[4767, 4962, 4927, 4465, 4559])
    print('')


if __name__ == '__main__':
    if platform == 'darwin':
        PROJECT_DIR = '/Users/kristijanbartol/LOOM/'
        SMPL_DIR = '/Users/kristijanbartol/data/smpl/models/'
    else:
        PROJECT_DIR = '/home/kristijan/LOOM/'
        SMPL_DIR = '/home/kristijan/data/smpl/models/'

    smpl_model = SMPL(
        model_path=os.path.join(SMPL_DIR, f'SMPL_FEMALE.pkl'), 
        gender='female'
    )
    body_verts = smpl_model(
        betas=torch.zeros((1, 10))
    ).vertices[0].cpu().detach().numpy()
    body_faces = smpl_model.faces

    cut_single_path(body_verts, body_faces)
