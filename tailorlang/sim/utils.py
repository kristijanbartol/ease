import os
import numpy as np
import trimesh
from copy import deepcopy
from smplx import SMPL
from plyfile import PlyData, PlyElement

from tailorlang import const
from tailorlang.const import apply_angle_offset


class ParamMeshUV():
    
    def __init__(self, mesh_3d_list, mesh_2d_list, garment_part):
        self.mesh_3d_with_duplicates = trimesh.util.concatenate(mesh_3d_list)
        self.mesh_3d = trimesh.Trimesh(
            vertices=self.mesh_3d_with_duplicates.vertices,
            faces=self.mesh_3d_with_duplicates.faces,
            process=True        # remove duplicates for the "active" vertices
        )
        self.pattern_2d_subdivided = self._process_pattern(mesh_2d_list, garment_part)
        self.duplicate_pairs_dict = self._find_duplicate_vertices(      # needed to update duplicate mesh before preparing for sim (transform + subdivide)
            mesh=self.mesh_3d,
            mesh_with_duplicates=self.mesh_3d_with_duplicates
        )
        self.duplicate_pairs_dict_subdivided = None
        self.mesh_3d_subdivided = None
        self.mesh_3d_with_duplicates_subdivided = None
        
    @staticmethod
    def _process_pattern(mesh_2d_list, garment_part):
        for mesh_idx in range(len(mesh_2d_list)):
            if garment_part == 'lower':
                mesh_2d_list[mesh_idx].vertices[:, 0] += mesh_idx * 1.0
            else:
                mesh_2d_list[mesh_idx].vertices[:, 1] += 1.0
                mesh_2d_list[mesh_idx].vertices[:, 0] += mesh_idx * 1.0
        return trimesh.util.concatenate(mesh_2d_list).subdivide()
    
    def update_duplicate_mesh(self):
        for duplicate_idx, non_duplicate_idx in self.duplicate_pairs_dict.items():
            self.mesh_3d_with_duplicates.vertices[duplicate_idx] = self.mesh_3d.vertices[non_duplicate_idx]
            
    def subdivide(self):
        self.mesh_3d_with_duplicates_subdivided = self.mesh_3d_with_duplicates.subdivide()
        self.mesh_3d_subdivided = trimesh.Trimesh(                  # used for simulation
            vertices=self.mesh_3d_with_duplicates_subdivided.vertices,
            faces=self.mesh_3d_with_duplicates_subdivided.faces,
            process=True
        )
        self.duplicate_pairs_dict_subdivided = self._find_duplicate_vertices(
            mesh=self.mesh_3d_subdivided,
            mesh_with_duplicates=self.mesh_3d_with_duplicates_subdivided
        )
        
    def update_duplicate_mesh_subdivided(self):
        # Update the subdivided duplicate mesh with the subdivided simulated mesh
        for duplicate_idx, non_duplicate_idx in self.duplicate_pairs_dict_subdivided.items():
            self.mesh_3d_with_duplicates_subdivided.vertices[duplicate_idx] = self.mesh_3d_subdivided.vertices[non_duplicate_idx]
        
    def export(self, output_path):
        # When storing non-skintight (for simulation), the mesh is transformed but we need an original size for rendering
        if self.mesh_3d_with_duplicates_subdivided.vertices.max() - self.mesh_3d_with_duplicates_subdivided.vertices.min() > 5.0:
            _retrans_mesh_3d_with_duplicates_subdivided = deepcopy(self.mesh_3d_with_duplicates_subdivided)
            _retrans_mesh_3d_with_duplicates_subdivided.apply_transform(trimesh.transformations.rotation_matrix(
                angle=-np.pi/2,
                direction=[1, 0, 0] 
            ))
            _retrans_mesh_3d_with_duplicates_subdivided.vertices /= 10.
            mesh_3d_with_duplicates_plydata = trimesh_to_plydata(_retrans_mesh_3d_with_duplicates_subdivided)  
        else:
            mesh_3d_with_duplicates_plydata = trimesh_to_plydata(self.mesh_3d_with_duplicates_subdivided)
        add_uv_coordinates(
            mesh_3d=mesh_3d_with_duplicates_plydata,
            uv_coordinates=self.pattern_2d_subdivided.vertices[:, :2],
            output_path=output_path
        )
        
    @staticmethod
    def _find_duplicate_vertices(mesh, mesh_with_duplicates):
        duplicate_groups = trimesh.grouping.group_rows(mesh_with_duplicates.vertices, digits=6)
        _duplicate_pairs_dict = {}
        for group in duplicate_groups:
            non_duplicate_index = mesh.kdtree.query(mesh_with_duplicates.vertices[group[0]])[1]
            for orig_index in group:
                _duplicate_pairs_dict[orig_index] = non_duplicate_index
        return _duplicate_pairs_dict


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


def process_base_for_simulation(param_mesh: ParamMeshUV):
    param_mesh.mesh_3d_with_duplicates.vertices *= 10.
    param_mesh.mesh_3d_with_duplicates.apply_transform(trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0] 
    ))
    param_mesh.subdivide()
    return param_mesh


def postprocess_base_after_simulation(
        param_mesh: ParamMeshUV, 
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
        upper_param_mesh: ParamMeshUV,         # upper trimesh mesh (merged)
        lower_param_mesh: ParamMeshUV,          # lower trimesh mesh (merged)
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
        lower_param_mesh.export(lower_path)
    else:
        upper_param_mesh.mesh_3d_subdivided.export(upper_path)
        lower_param_mesh.mesh_3d_subdivided.export(lower_path)
        upper_param_mesh.export(upper_path_with_uv)     # mesh with UV and duplicates for mid-result and rendering
        lower_param_mesh.export(lower_path_with_uv)     # mesh with UV and duplicates for mid-result and rendering
    
    return upper_path, lower_path


def update_meshes_after_simulation(
        sim_dir, 
        base_param_mesh_dict
    ):
    # Update simulation results to include uv coordinates as well + apply inverse transformations
    upper_mesh = trimesh.load(os.path.join(sim_dir, 'base_upper.ply'))
    lower_mesh = trimesh.load(os.path.join(sim_dir, 'base_lower.ply'))
    
    postprocess_base_after_simulation(
        param_mesh=base_param_mesh_dict['upper'], 
        sim_mesh=upper_mesh, 
        output_path=os.path.join(sim_dir, 'base_upper_uv.ply')
    )
    postprocess_base_after_simulation(
        param_mesh=base_param_mesh_dict['lower'], 
        sim_mesh=lower_mesh, 
        output_path=os.path.join(sim_dir, 'base_lower_uv.ply')
    )
