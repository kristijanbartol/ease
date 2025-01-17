import os
import numpy as np
import trimesh
from smplx import SMPL
from plyfile import PlyData, PlyElement

from tailorlang import const
from tailorlang.const import apply_angle_offset


class ParamMesh():
    
    def __init__(self, mesh_3d_list, mesh_2d_list, faces):
        self.mesh_3d_with_duplicates = trimesh.Trimesh(
            vertices=np.vstack([x.vertices for x in mesh_3d_list]),
            faces=faces,
            process=False
        )
        self.faces = faces
        self.mesh_3d = trimesh.Trimesh(
            vertices=self.mesh_3d_with_duplicates.vertices,
            faces=faces,
            process=True        # remove duplicates for the "active" vertices
        )
        self.pattern_2d = trimesh.Trimesh(
            vertices=np.vstack([x.vertices for x in mesh_2d_list]),
            faces=faces
        )
        self.pattern_2d_subdivided = self._subdivide_patches(mesh_2d_list)
        self.duplicate_pairs_dict = self._find_duplicate_vertices()
        self.verts_3d_subdivided = None
        self.verts_3d_with_duplicates_subdivided = None

    def _subdivide_patches(self, mesh_2d_list):
        subdivided_patches = []
        for patch in mesh_2d_list:
            subdivided_patches.append(patch.subdivide())
        _pattern_2d_subdivided = trimesh.Trimesh(
            vertices=np.vstack([x.vertices for x in subdivided_patches]),
            faces=self.faces,
            process=False
        )
        return _pattern_2d_subdivided

    def _find_duplicate_vertices(self, tolerance=1e-6):
        duplicate_groups = trimesh.grouping.group_rows(self.mesh_3d_with_duplicates.vertices, digits=6)
        _duplicate_pairs_dict = {}
        for group in duplicate_groups:
            non_duplicate_index = self.mesh_3d.kdtree.query(self.mesh_3d_with_duplicates.vertices[group[0]])[1]
            for orig_index in group:
                _duplicate_pairs_dict[orig_index] = non_duplicate_index
        return _duplicate_pairs_dict
    
    def update_duplicate_mesh(self):
        for duplicate_idx, non_duplicate_idx in self.duplicate_pairs_dict.items():
            self.mesh_3d_with_duplicates.vertices[duplicate_idx] = self.mesh_3d.vertices[non_duplicate_idx]

    def subdivide(self):
        self.verts_3d_subdivided = self.mesh_3d_with_duplicates.subdivide()
        self.verts_3d_with_duplicates_subdivided = self.mesh_3d_with_duplicates.subdivide()
        
    def export(self, output_path, subdivided=True):
        if subdivided:
            mesh_3d_with_duplicates_subdivided = self.mesh_3d_with_duplicates.subdivide()
            mesh_3d_with_duplicates_plydata = trimesh_to_plydata(mesh_3d_with_duplicates_subdivided)
            add_uv_coordinates(
                mesh_3d=mesh_3d_with_duplicates_plydata,
                uv_coordinates=self.pattern_2d_subdivided.vertices[:, :2],
                output_path=output_path
            )
        else:
            mesh_3d_with_duplicates_plydata = trimesh_to_plydata(self.mesh_3d_with_duplicates)
            add_uv_coordinates(
                mesh_3d=mesh_3d_with_duplicates_plydata,
                uv_coordinates=self.pattern_2d.vertices[:, :2],
                output_path=output_path
            )


def trimesh_to_plydata(trimesh_mesh):
    # Extract vertices and faces
    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces

    # Prepare vertices as a structured array
    vertex_array = np.array(
        [(v[0], v[1], v[2]) for v in vertices],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )

    # Prepare faces as a structured array
    face_array = np.array(
        [(list(f),) for f in faces],
        dtype=[('vertex_indices', 'i4', (3,))]
    )

    # Create PlyElement objects for vertices and faces
    vertex_element = PlyElement.describe(vertex_array, 'vertex')
    face_element = PlyElement.describe(face_array, 'face')

    # Create and return a PlyData object
    ply_data = PlyData([vertex_element, face_element], text=True)
    return ply_data


def add_uv_coordinates(mesh_3d, uv_coordinates, output_path):        
    # Get vertex data from the original mesh
    vertex_data = mesh_3d['vertex']
    
    # Create a new vertex element with UV coordinates
    vertex_dtype = vertex_data.data.dtype.descr + [('s', 'f4'), ('t', 'f4')]
    
    # Create new vertex array with all properties
    new_vertex_data = np.empty(len(vertex_data.data), dtype=vertex_dtype)
    
    # Copy existing properties
    for prop in vertex_data.data.dtype.names:
        new_vertex_data[prop] = vertex_data.data[prop]
    
    # Add UV coordinates
    new_vertex_data['s'] = uv_coordinates[:, 0]
    new_vertex_data['t'] = uv_coordinates[:, 1]
    
    # Create new vertex element
    vertex_element = PlyElement.describe(new_vertex_data, 'vertex')
    
    # Create new PLY data with updated vertices and original faces
    new_plydata = PlyData([vertex_element, mesh_3d['face']], text=True)
    
    # Write to new file
    new_plydata.write(output_path)


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


def process_base_for_simulation(param_mesh):
    param_mesh.mesh_3d.vertices *= 10.
    param_mesh.mesh_3d.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0] 
    ))
    param_mesh.update_duplicate_mesh()
    param_mesh.subdivide()
    return param_mesh


def postprocess_base_after_simulation(param_mesh, sim_mesh, output_path):
    param_mesh.mesh_3d.vertices = sim_mesh.vertices
    param_mesh.mesh_3d.apply_transform(trimesh.transformations.rotation_matrix(
        angle=-np.pi/2,
        direction=[1, 0, 0] 
    ))
    param_mesh.mesh_3d.vertices /= 10.
    param_mesh.update_duplicate_mesh()
    param_mesh.export(output_path)


def process_refit_for_simulation(garment_mesh):
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
        refit_pose,         # the pose to which to refit, 'base' in case of not processing the refit pose
        body_mesh,          # target-01.ply -> transformed
        upper_param_mesh,         # upper trimesh mesh (merged)
        lower_param_mesh,          # lower trimesh mesh (merged)
    ):
    non_skintight_garment_dir = f'results/non-skintight/{experiment_name}'
    os.makedirs(non_skintight_garment_dir, exist_ok=True)
    
    body_mesh.export(body_path)
    
    upper_path = os.path.join(non_skintight_garment_dir, f'{refit_pose}_upper.ply')
    lower_path = os.path.join(non_skintight_garment_dir, f'{refit_pose}_lower.ply')
    upper_path_with_uv = os.path.join(non_skintight_garment_dir, f'{refit_pose}_upper_uv.ply')
    lower_path_with_uv = os.path.join(non_skintight_garment_dir, f'{refit_pose}_lower_uv.ply')
    
    if refit_pose == 'base':
        upper_param_mesh.mesh_3d.export(upper_path)
        lower_param_mesh.mesh_3d.export(lower_path)
        upper_param_mesh.export(upper_path_with_uv)     # mesh with UV and duplicates
        lower_param_mesh.export(lower_path_with_uv)     # mesh with UV and duplicates
    else:
        upper_param_mesh.export(upper_path)
        lower_param_mesh.export(lower_path)
    
    return upper_path, lower_path


def update_meshes_after_simulation(base_param_mesh_dict):
    # Update simulation results to include uv coordinates as well + apply inverse transformations
    shirt_mesh = trimesh.load('results/sim/base_shirt.ply')
    pant_mesh = trimesh.load('results/sim/base_pant.ply')
    postprocess_base_after_simulation(base_param_mesh_dict['upper'], shirt_mesh, 'results/sim/base_shirt_uv.ply')
    postprocess_base_after_simulation(base_param_mesh_dict['lower'], pant_mesh, 'results/sim/base_pant_uv.ply')
    