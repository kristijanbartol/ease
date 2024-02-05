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


def project_point_onto_line_segment(line_point1, line_point2, point):
    # Line direction vector
    line_dir = line_point2 - line_point1
    line_length = np.linalg.norm(line_dir)
    line_dir_normalized = line_dir / line_length

    # Vector from one line point to the point
    vec_to_point = point - line_point1

    # Project vec_to_point onto line_dir_normalized
    projection_length = np.dot(vec_to_point, line_dir_normalized)
    
    # Clamp the projection length to the line segment
    projection_length_clamped = max(0, min(line_length, projection_length))
    projection = line_point1 + projection_length_clamped * line_dir_normalized

    return projection


def get_direction_vector(point1, point2, normalize=True):
    # Calculate the direction vector from point1 to point2
    direction_vector = point2 - point1
    
    # Optionally normalize the vector to have unit length
    if normalize:
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
    
    return direction_vector


def get_local_yarn_direction(
        mesh,
        point,
        global_yarn
    ):
    _, _, face_index = mesh.nearest.on_surface(point)
    point_normal = mesh.face_normals[face_index]
    local_yarn = global_yarn - (np.dot(point_normal, global_yarn) * point_normal)
    return local_yarn


def compare_local_position(point1, point2, ref_point, local_yarn):
    left_direction_normalized = (-local_yarn) / np.linalg.norm(local_yarn)
    
    vec_to_point1 = point1 - ref_point
    vec_to_point2 = point2 - ref_point
    
    projection1 = np.dot(vec_to_point1, left_direction_normalized)
    projection2 = np.dot(vec_to_point2, left_direction_normalized)

    if projection1 <= projection2:  # point1 (more left/down or equal to) point2
        return True
    elif projection1 > projection2: # point1 (more right/up to) point2
        return False


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

    # List of inner faces: to determine when out of boundaries.
    # List of boundary points: for traversing the boundary.
    # Dictionary of {face: (lbound, rbound)}: to get the corresponding boundary points given a face.
    mesh, boundary_face_points_dict, boundary_points, inner_faces = prepare_grid_processing_data(
        orig_verts=smpl_model().vertices[0].cpu().detach().numpy(),
        orig_faces=smpl_model.faces
    )

    mesh.rezero()   # recalculate normals
    GLOBAL_WARP = np.array([0, 0, 1])
    GLOBAL_WEFT = np.array([1, 0, 0])
    DISCRETE_STEP = 0.001
    LINE_DIST = 0.005
    NUM_NEXT = 3

    # TODO: Update the algorithm to also traverse the right direction.
    # TODO: Update the algorithm to also include the weft-warp direction.

    warp_lines_dict = {}
    boundary_idx = 0
    current_line_point = boundary_points[boundary_idx]
    while True:     # inner while-else is breaking the outer loop
        warp_lines_dict[current_line_point] = [current_line_point]
        inside_boundaries = True

        while inside_boundaries:
            local_warp = get_local_yarn_direction(
                mesh=mesh,
                point=warp_lines_dict[current_line_point][-1],
                global_yarn=GLOBAL_WARP
            )
            next_point = warp_lines_dict[current_line_point][-1] + local_warp * DISCRETE_STEP
            next_point, _, face = trimesh.proximity.closest_point(
                mesh, 
                next_point
            )
            if face not in inner_faces:
                inside_boundaries = False
                last_point = warp_lines_dict[current_line_point][-1]
                while face not in boundary_face_points_dict:
                    direction_vector = get_direction_vector(
                        last_point,
                        next_point
                    )
                    last_point += direction_vector * (DISCRETE_STEP / 10)
                    last_point, _, face = trimesh.proximity.closest_point(
                        mesh, 
                        last_point
                    )
                left_boundary, right_boundary = boundary_face_points_dict[face]
                last_point = project_point_onto_line_segment(
                    left_boundary,
                    right_boundary,
                    last_point
                )
                warp_lines_dict[current_line_point].append(last_point)
            else:
                warp_lines_dict[current_line_point].append(next_point)

        # Calculate weft direction on the local tangent of the current line point.
        local_weft = get_local_yarn_direction(
            mesh=mesh,
            point=current_line_point,
            global_yarn=GLOBAL_WEFT
        )
        local_next_line_point = current_line_point - local_weft * LINE_DIST
        local_boundary_points = trimesh.proximity.closest_point(
            mesh, 
            boundary_points
        )
        # Iterate through boundary points projected to local plane to search for the
        # next line point. If not found, the loop will "naturally" terminate.
        while boundary_idx < boundary_points.shape[0]:
            boundary_idx += 1
            if compare_local_position(
                point1=local_boundary_points[boundary_idx],
                point2=local_next_line_point,
                ref_point=current_line_point,
                local_yarn=local_weft
            ):
                # If the local boundary point is more toward the current direction,
                # then choose its corresponding original boundary point as the
                # next current line point.
                # NOTE: This generally sets the next line point to be more in the
                # current direction, which will be detected by the optimization and
                # moved back to prevent stretching.
                current_line_point = boundary_points[boundary_idx]
        else:
            break
