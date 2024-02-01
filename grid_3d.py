from smplx import SMPL
import argparse
import trimesh
import numpy as np

from const import (
    KEYPOINTS,
    PANT_LENGTH,
    SHIRT_LENGTH,
    SLEEVE_LENGTH
)
from garment import Garment
from geometry import (
    find_init_face,
    project_boundaries,
    subdivide_mesh,
    extract_boundaries
)


def prepare_grid_processing_data(
        orig_verts, 
        orig_faces
    ):
    verts, faces = subdivide_mesh(
        verts=orig_verts, 
        faces=orig_faces
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    boundary_points = extract_boundaries(
        args=args,
        orig_verts=orig_verts,
        sub_verts=verts,
        body_part_keypoints=KEYPOINTS['upper_front'],
        num_points=1000
    )
    boundary_face_to_points_dict, boundary_points = project_boundaries(
        mesh=mesh,
        points=boundary_points
    )
    starting_face = find_init_face(
        mesh=mesh, 
        start_point=boundary_face_to_points_dict.values()
    )
    inner_faces = Garment(verts, faces).select_faces(
        boundary_faces=boundary_face_to_points_dict.keys(),
        starting_face=starting_face
    )
    return (
        mesh,
        boundary_face_to_points_dict,
        boundary_points,
        inner_faces
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gender', '-G', type=str, choices=['male', 'female', 'neutral'], default='female')
    parser.add_argument('--subdivide', dest='subdivide', action='store_true')
    parser.add_argument('--shirt_length', '-S', type=float, default=SHIRT_LENGTH)
    parser.add_argument('--pant_length', '-P', type=float, default=PANT_LENGTH)
    parser.add_argument('--sleeve_length', '-L', type=float, default=SLEEVE_LENGTH)
    args = parser.parse_args()

    smpl_model = SMPL(model_path='/data/hierprob3d/smpl/SMPL_FEMALE.pkl')

    orig_verts = smpl_model().vertices[0].cpu().detach().numpy()
    orig_faces = smpl_model.faces

    mesh, boundary_face_points_dict, boundary_points, inner_faces = prepare_grid_processing_data(
        orig_verts=smpl_model().vertices[0].cpu().detach().numpy(),
        orig_faces=smpl_model.faces
    )

    DISCRETE_STEP = 0.001
    current_point = boundary_points[0]
    inside_boundaries = True
    mesh.rezero()   # recalculate normals
    global_warp = np.array([0, 0, 1])
    global_weft = np.array([1, 0, 0])
    while inside_boundaries:
        _, _, face_index = mesh.nearest.on_surface(current_point)
        point_normal = mesh.face_normals[face_index]
        local_warp = global_warp - (np.dot(point_normal, global_warp) * point_normal)
        next_point = current_point + local_warp * DISCRETE_STEP
        next_point, _, face = trimesh.proximity.closest_point(mesh, next_point)
        if face in boundary_face_points_dict:
            # TODO: continue
            pass
        current_point = next_point
