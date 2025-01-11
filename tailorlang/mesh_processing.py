from collections import defaultdict
from typing import Tuple, List
import os
import shutil
import torch
import numpy as np
from smplx import SMPL
import json
from smplx import SMPL
import trimesh
from scipy.spatial import KDTree

from tailorlang import const
from tailorlang.const import (
    SMPL_DIR,
    COMPONENT_SIGN_DICT,
    DART_ORIENTS,
    DISPLACEMENTS,
    INIT_IDXS,
    INIT_UPPER_FRONT,
    INIT_UPPER_BACK,
    INIT_FRONT_RIGHT_SLEEVE,
    INIT_FRONT_LEFT_SLEEVE,
    INIT_BACK_LEFT_SLEEVE,
    INIT_BACK_RIGHT_SLEEVE,
    INIT_LEFT_BACK_PANT,
    INIT_RIGHT_BACK_PANT,
    INIT_LEFT_FRONT_PANT,
    INIT_RIGHT_FRONT_PANT,
    PRE_SEAMS_DICT,
    FIXED_POINTS_DICT,
    PATCH_LIST,
    PLANE_ORIENT_DICT,
    SEGMENT_TO_ID,
    SEGMENT_TO_SEAMLINES_DICT,
    SEAM_TO_SEAM_IDX_DICT,
    SEGMENT_TO_DARTS,
    PATCH_TO_PRE_SEAMS_DICT
)
from tailorlang.garment import (
    BodySet,
    Garment,
    DesignParameters
)
from tailorlang.geometry import (
    modify_mesh_with_plane_cut,
    point_side,
    update_face
)
from tailorlang.io import (
    export,
    export_body_mesh,
    load_preselected,
    save_darts_files,
    save_seamline_pairs_file,
    store_preselected
)
from tailorlang.pybind import apply_remesh
from tailorlang.submodules import run_parameterization
from tailorlang.geo.dart_extractor import select_faces_in_dart



from tailorlang.eval.stretch_utils import color_code_stretches


class MeshProcessor:
    
    @staticmethod
    def build_vertex_adjacency_list(F):
        adjacency_list = {}
        for triangle in F:
            for vertex in triangle:
                if vertex not in adjacency_list:
                    adjacency_list[vertex] = set()
                # Add the neighboring vertices
                adjacency_list[vertex].update(triangle)
                adjacency_list[vertex].remove(vertex)  # A vertex is not a neighbor to itself
        return adjacency_list
    
    @staticmethod
    def _map_vertex_idx_to_face_idxs(faces):
        # Create a map of vertex to faces for quick lookup
        # TODO: Improve this to avoid creating a map for each dart.
        vertex_idx_to_face_idxs = defaultdict(list)
        for i, face in enumerate(faces):
            for v in face:
                vertex_idx_to_face_idxs[v].append(i)
        return vertex_idx_to_face_idxs
    
    @staticmethod
    def update_homologous_mesh(smpl_dir, target_gender, barycentric_coords):
        """
        Updates vertices of a homologous mesh using barycentric coordinates obtained from a previous mesh modification.
        
        Parameters:
            vertices_target: numpy array of vertex positions for the target mesh to be updated
            faces: face indices shared between the meshes
            barycentric_coords: dictionary mapping vertex indices to their barycentric coordinates
            
        Returns:
            numpy array of updated vertex positions for the target mesh
        """
        target_smpl_model = SMPL(model_path=os.path.join(smpl_dir, f'SMPL_{target_gender.upper()}.pkl'), 
             gender=target_gender,
        )
        orig_target_verts = target_smpl_model.vertices[0].cpu().detach().numpy()
        new_verts = orig_target_verts.copy()
        faces = target_smpl_model.faces
        
        # Update each vertex that has barycentric coordinates
        for vertex_idx, bary_coords in barycentric_coords.items():
            # Get the face that contains this vertex
            containing_face = None
            for face in faces:
                if vertex_idx in face:
                    containing_face = face
                    break
                    
            if containing_face is None:
                continue
                
            # Get the triangle vertices from the target mesh
            triangle_vertices = orig_target_verts[containing_face]
            
            # Calculate the new position using barycentric coordinates
            new_position = (bary_coords[0] * triangle_vertices[0] + 
                        bary_coords[1] * triangle_vertices[1] + 
                        bary_coords[2] * triangle_vertices[2])
            
            # Update the vertex position
            new_verts[vertex_idx] = new_position
            
        return new_verts
    
    @staticmethod
    def create_dart(patch_verts, patch_faces, selected_vidxs, dart_orient):
        # New vertex idx map contains the map (orig_selected_idx->new_vertex_idx).
        new_vertex_idx_map = {v_idx: len(patch_verts) + idx for idx, v_idx in enumerate(selected_vidxs[:-1])}
        # Fast lookup of the corresponding face idxs, given vertex idx.
        vertex_idx_to_face_idxs = MeshProcessor._map_vertex_idx_to_face_idxs(patch_faces)
        updated_faces = patch_faces.copy()
        
        # Do not process the last edge, containing the dart tip.
        for idx, _ in enumerate(selected_vidxs[:-2]):
            v1_idx, v2_idx = selected_vidxs[idx], selected_vidxs[idx+1]
            edge = patch_verts[v1_idx], patch_verts[v2_idx]
            adjacent_face_idxs = [f for f in vertex_idx_to_face_idxs[v1_idx]]
            
            for face_idx in adjacent_face_idxs:
                face = patch_faces[face_idx]
                center = patch_verts[face].mean(axis=0)
                side = point_side(center, edge, dart_orient)

                # For faces "above" the dart, replace the old with new, added vertex idxs.
                if side > 0:
                    updated_face = face.copy()
                    # Iterate through vertex indices of the face and update if selected.
                    for v_idx in face:
                        if v_idx in new_vertex_idx_map:
                            # Update face the number of times v_idx is in the selected set.
                            updated_face = update_face(
                                face=updated_face, 
                                v_idx_to_update=v_idx, 
                                new_v_idx=new_vertex_idx_map[v_idx]
                            )
                    # Finally, replace the original face with the updated face. 
                    updated_faces[face_idx] = updated_face
            
        # Add new vertices (all dart vertices except the tip (last vertex)). 
        new_vertices = patch_verts[selected_vidxs[:-1]]
        #new_vertices[:, 1] += 0.00001   # add tiny displacement so that trimesh doesn't skip "duplicate" vertices
        new_vertices[:, 1] += 0.001   # add tiny displacement so that trimesh doesn't skip "duplicate" vertices
        updated_vertices = np.vstack((patch_verts, new_vertices))
        
        # Create a dart vertex pair strategy relevant for further processing (parameterization).
        dart_pairs = [selected_vidxs[-1]]
        for i in range(len(selected_vidxs) - 2, -1, -1):
            dart_pairs.append((selected_vidxs[i], new_vertex_idx_map[selected_vidxs[i]]))

        return updated_vertices, updated_faces, dart_pairs
    
    @staticmethod
    def compute_triangle_directions(vertices: np.ndarray, faces: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute warp and weft directions and unit distance points for each triangle in a mesh.
        
        Args:
            vertices: (V, 3) array of vertex coordinates
            faces: (F, 3) array of face indices
        
        Returns:
            Tuple containing:
            - List of barycentric coordinates for C_x points
            - List of barycentric coordinates for C_y points
        """
        UNIT_DISTANCE = 1.0
        c_x_coords = []
        c_y_coords = []
        
        def compute_barycentric(p: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
            """
            Compute barycentric coordinates using area method.
            """
            # Compute vectors
            v0 = v2 - v1
            v1_vec = v3 - v1
            v2_vec = p - v1
            
            # Compute dot products
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1_vec)
            d11 = np.dot(v1_vec, v1_vec)
            d20 = np.dot(v2_vec, v0)
            d21 = np.dot(v2_vec, v1_vec)
            
            # Compute barycentric coordinates
            denom = d00 * d11 - d01 * d01
            beta = (d11 * d20 - d01 * d21) / denom
            gamma = (d00 * d21 - d01 * d20) / denom
            alpha = 1.0 - beta - gamma
            
            return np.array([alpha, beta, gamma])
        
        for face in faces:
            # Get triangle vertices
            v1, v2, v3 = vertices[face]
            
            # Compute triangle normal
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            
            # Project global X and Y axes onto triangle plane
            global_x = np.array([1.0, 0.0, 0.0])
            global_y = np.array([0.0, 1.0, 0.0])
            
            weft = global_x - np.dot(global_x, normal) * normal
            warp = global_y - np.dot(global_y, normal) * normal
            
            # Normalize projected directions
            weft = weft / np.linalg.norm(weft)
            warp = warp / np.linalg.norm(warp)
            
            # Compute triangle centroid in world coordinates
            centroid = (v1 + v2 + v3) / 3.0
            
            # Compute unit distance points
            c_x = centroid + weft * UNIT_DISTANCE
            c_y = centroid + warp * UNIT_DISTANCE
            
            # Compute barycentric coordinates
            c_x_coords.append(compute_barycentric(c_x, v1, v2, v3))
            c_y_coords.append(compute_barycentric(c_y, v1, v2, v3))
            
        return np.stack((c_x_coords, c_y_coords), axis=1)
    
    @staticmethod
    def extract_local_stretches(verts, faces, design_params, patch_label, side=None) -> np.ndarray:
        if 'sleeve' in patch_label:
            stretches_u = np.ones(faces.shape[0])   # Do not design elongation direction (only looseness)
            stretches_v = np.ones(faces.shape[0]) * design_params.sleeve_looseness
        else:
            if 'upper' in patch_label:
                stretches_u = np.ones(faces.shape[0]) * design_params.shirt_looseness
            else:
                stretches_u = np.ones(faces.shape[0]) * design_params.pant_looseness
            stretches_v = np.ones(faces.shape[0])   # Do not design elongation direction (only looseness)

        # TODO: Implement other stretch functions using design_params.
        
        return {
            'u': stretches_u,
            'v': stretches_v
        }


class MeshTopologyTransfer:
    def __init__(self, source_vertices, source_faces, target_vertices):
        """
        Initialize the topology transfer between source (remeshed) and target meshes.
        
        Parameters:
        source_vertices: numpy array (Nx3) containing vertex coordinates of source mesh
        source_faces: numpy array (Mx3) containing face indices of source mesh
        target_vertices: numpy array (Nx3) containing vertex coordinates of target mesh
        """
        self.source_vertices = source_vertices
        self.source_faces = source_faces
        self.target_vertices = target_vertices
        self.correspondence_map = None
        
    def compute_correspondence(self):
        """
        Compute correspondence between source and target mesh vertices using KD-tree.
        Returns mapping from source vertices to target vertices.
        """
        target_tree = KDTree(self.target_vertices)
        distances, indices = target_tree.query(self.source_vertices)
        self.correspondence_map = indices
        return indices
    
    def transfer_topology(self):
        """
        Transfer the remeshed topology from source to target mesh.
        Returns tuple of (vertices, faces) for the new mesh.
        """
        if self.correspondence_map is None:
            self.compute_correspondence()
            
        new_vertices = self.target_vertices[self.correspondence_map]
        return new_vertices, self.source_faces
    
    def compute_transfer_error(self):
        """
        Compute the geometric error of the topology transfer.
        Returns mean and max distance between corresponding vertices.
        """
        if self.correspondence_map is None:
            self.compute_correspondence()
            
        mapped_vertices = self.target_vertices[self.correspondence_map]
        distances = np.linalg.norm(self.source_vertices - mapped_vertices, axis=1)
        return np.mean(distances), np.max(distances)

def transfer_remesh_to_target(source_vertices, source_faces, target_vertices_list):
    """
    Apply remeshing from a source mesh to a set of homologous meshes.
    
    Parameters:
    source_vertices: numpy array (Nx3) of remeshed source vertex positions
    source_faces: numpy array (Mx3) of remeshed source face indices
    target_vertices_list: list of numpy arrays (Nx3) containing vertex positions of target meshes
    
    Returns:
    list of tuples (vertices, faces) for each remeshed target mesh
    """
    remeshed_meshes = []
    
    for target_vertices in target_vertices_list:
        transfer = MeshTopologyTransfer(source_vertices, source_faces, target_vertices)
        remeshed_vertices, remeshed_faces = transfer.transfer_topology()
        remeshed_meshes.append((remeshed_vertices, remeshed_faces))
        
    return remeshed_meshes


class MeshState:
    
    def __init__(self, config):
        self.config = config
        self.use_darts = config.use_darts
        self.apply_remesh = config.apply_remesh
        
        # Patch-threshold dictionaries
        self.threshold_dict = {}
        
        # Patch-to-vertex-indices dictionary
        self.full_patch_idxs_dict = {}
        self.masked_patch_idxs_dict = {}
        
        self._init_static(config.body_set)
        self._prepare_init()
        
        # Data structures for the finalization
        # TODO: Have one reference body and a set of target bodies (not necessarily including the reference body)
        self.ref_patch_verts_dict = {}
        self.ref_patch_faces_dict = {}
        self.target_patch_verts_dict_list = [{} for _ in range(self.body_set.num_targets)]
        self.target_patch_faces_dict_list = [{} for _ in range(self.body_set.num_targets)]
        
        self.old_to_new_idx_dict = {}
        
        self.seam_to_segment_vertex_pairs = {}
        
        self.dart_dict = {}
        
        self.uv_unit_directions_dict = {}
        self.local_stretches_dict = {}
        
    def _init_static(self, body_set):
        with open(f'config/designs/long.json', 'r') as json_file:
            init_design_dict = json.load(json_file)
        self.design_params = DesignParameters(init_design_dict)
        
        with open(f'config/body_sets/{body_set}.json', 'r') as json_file:
            set_dict = json.load(json_file)
        self.body_set = BodySet(set_dict)
        self.ref_gender = self.body_set.ref['gender']
        ref_shape_fun = getattr(const, self.body_set.ref['shape'])
        
        self.smpl_model = SMPL(
            model_path=os.path.join(SMPL_DIR, f'SMPL_{self.ref_gender.upper()}.pkl'), 
            gender=self.ref_gender
        )
        
        self.canonical_mesh = {
            'vertices': self.smpl_model(
                betas=ref_shape_fun()
            ).vertices[0].cpu().detach().numpy(),
            'faces': self.smpl_model.faces,
            'color': (0.7, 0.7, 0.7, 1.0)  # Gray
        }
        self.active_mesh = self.canonical_mesh.copy()   # set immediately (since the initial design is applied)
        self.bary_coords_for_active_cuts_dict = {}      # component: bary_coords
        
        self.target_bodies_verts = []
        
        # Patch-to-(pre-)seamlines dictionary
        # TODO: Seamlines should also become seamline INDICES!
        # NOTE: ref_seamlines_dict and masked_seamlines_dict not needed at the moment (since old_to_new_dict finally selects the seamlines anyway)
        self.ref_seamlines_dict = {}
        for seamline_label in PRE_SEAMS_DICT:
            vertex_idxs = PRE_SEAMS_DICT[seamline_label]
            verts = [self.canonical_mesh['vertices'][idx] for idx in vertex_idxs]
            self.ref_seamlines_dict[seamline_label] = np.array(verts)
        
    def _prepare_init(self):
        # TODO: Preload these for efficiency (for each already-prepared reference body)
        self.apply_length_params()
        self.apply_pre_seams()
        self.apply_masks()
    
    def flood_fill(self, vertex_positions, boundary_vertices, start_vertex, threshold_params):
        """
        Unified flood fill function that can handle both vertical (pants) and horizontal (sleeve) filling.
        
        Args:
            vertex_positions: Array of vertex positions
            boundary_vertices: List of boundary vertex indices
            start_vertex: Starting vertex index for flood fill
            threshold_params: Dictionary containing threshold parameters
                For vertical (pants) fill:
                    {'mode': 'vertical', 'threshold': y_threshold}
                For horizontal (sleeve) fill:
                    {'mode': 'horizontal', 'threshold': x_threshold, 'side': 'left'/'right'}
        
        Returns:
            List of selected vertex indices
        """
        def check_threshold(vertex_idx):
            if threshold_params['mode'] == 'vertical':
                return vertex_positions[vertex_idx, 1] >= threshold_params['threshold']
            else:  # horizontal mode
                x_coord = vertex_positions[vertex_idx, 0]
                if threshold_params['side'] == 'right':
                    return x_coord >= threshold_params['threshold']
                return x_coord <= threshold_params['threshold']

        # Convert boundary vertices to a set for efficient lookup
        boundary_set = set(boundary_vertices)

        # Initialize the stack with the start vertex
        stack = [start_vertex]
        visited = set()
        selected_vertices = set()

        while stack:
            vertex_idx = stack.pop()

            # Check if vertex meets all conditions
            if vertex_idx not in visited and vertex_idx not in boundary_set and check_threshold(vertex_idx):
                visited.add(vertex_idx)
                selected_vertices.add(vertex_idx)

                # Add unvisited neighbors to stack
                for neighbor_idx in self.vertex_adjacency_list[vertex_idx]:
                    if neighbor_idx not in visited:
                        stack.append(neighbor_idx)

        # Add boundary vertices that meet threshold conditions
        thresh_boundaries = [index for index in boundary_vertices if check_threshold(index)]
        selected_vertices.update(thresh_boundaries)

        return list(selected_vertices)
    
    def _get_threshold(self, init_fixed_vertex, component_sign, plane_orient, component_length):
        threshold_point = init_fixed_vertex + component_sign * component_length
        if plane_orient == 'horizontal':
            return threshold_point[1]
        else:
            return threshold_point[0]
    
    def apply_length_params(self):
        ''' Updates active_mesh and threshold_dict. '''
        component_lengths_dict = {
            'upper': self.design_params.shirt_length, 
            'sleeve': self.design_params.sleeve_length,
            'lower': self.design_params.pant_length
        }
        self.active_mesh['vertices'] = self.canonical_mesh['vertices']
        for component_label in ['upper', 'sleeve_left', 'sleeve_right', 'lower']:
            component_length = component_lengths_dict[component_label.split('_')[0]]
            self.threshold_dict[component_label] = self._get_threshold(
                init_fixed_vertex=self.canonical_mesh['vertices'][FIXED_POINTS_DICT[component_label]],
                component_sign=COMPONENT_SIGN_DICT[component_label],
                plane_orient=PLANE_ORIENT_DICT[component_label],
                component_length=component_length
            )
            #if not(self.ref_gender == 'male' and self.body_set.num_targets > 0):    # NOTE: SMPL doesn't work with v_template argument for males
            self.active_mesh['vertices'], self.bary_coords_for_active_cuts_dict[component_label] = modify_mesh_with_plane_cut(
                vertices=self.active_mesh['vertices'],
                faces=self.active_mesh['faces'],
                cutting_point=self.threshold_dict[component_label],
                plane_orientation=PLANE_ORIENT_DICT[component_label],
                sleeve_side=component_label.split('_')[1] if component_label[:6] == 'sleeve' else None
            )
                
        '''
        if not(self.ref_gender == 'male' and self.body_set.num_targets > 0):
            active_smpl = SMPL(
                model_path=os.path.join(SMPL_DIR, f'SMPL_{self.ref_gender.upper()}.pkl'), 
                gender=self.ref_gender,
                v_template=torch.from_numpy(self.active_mesh['vertices']).float()
            )
            self.active_mesh['vertices'] = active_smpl(
                body_pose=getattr(const, self.body_set.ref['pose'])(),
                betas=getattr(const, self.body_set.ref['shape'])(),
            ).vertices[0].cpu().detach().numpy()
        else:   # posed and shaped mesh (reference) but without a cut
            self.active_mesh['vertices'] = self.smpl_model(
                body_pose=getattr(const, self.body_set.ref['pose']),
                betas=getattr(const, self.body_set.ref['shape']),
            ).vertices[0].cpu().detach().numpy()
        '''
                
        trimesh.Trimesh(vertices=self.active_mesh['vertices'], faces=self.active_mesh['faces']).export('cut_mesh.ply')
        trimesh.Trimesh(vertices=self.canonical_mesh['vertices'], faces=self.canonical_mesh['faces']).export('canonical_mesh.ply')
    
    def apply_pre_seams(self):
        ''' Updates patch_idxs_dict. '''
        self.vertex_adjacency_list = MeshProcessor.build_vertex_adjacency_list(self.canonical_mesh['faces'])
        
        # Collect individual garment patches (initial).
        for patch_label in INIT_IDXS:
            patch_type = patch_label.split('_')[0]
            if patch_type == 'sleeve':
                sleeve_side = patch_label.split('_')[-1]
                threshold_key = f'{patch_type}_{sleeve_side}'
                params = {
                    'mode': 'horizontal',
                    'threshold': self.threshold_dict[threshold_key],
                    'side': sleeve_side
                }
            else:
                threshold_key = patch_type
                params = {
                    'mode': 'vertical',
                    'threshold': self.threshold_dict[patch_type]
                }
                
            patch_vert_idxs = self.flood_fill(
                vertex_positions=self.active_mesh['vertices'],
                boundary_vertices=sum(PATCH_TO_PRE_SEAMS_DICT[patch_label].values(), []),
                start_vertex=INIT_IDXS[patch_label],
                threshold_params=params
            )
            self.full_patch_idxs_dict[patch_label] = np.array(patch_vert_idxs, dtype=np.int16)
            
        self.full_patch_idxs_dict['upper'] = np.concatenate([idxs for key, idxs in self.full_patch_idxs_dict.items() if 'lower' not in key])
        self.full_patch_idxs_dict['lower'] = np.concatenate([idxs for key, idxs in self.full_patch_idxs_dict.items() if 'lower' in key])
        
    def apply_masks(self):
        ''' Applied when updating the garment parameters. '''
    
        def _upper_mask(verts, threshold_dict):
            return (verts[:, 1] >= threshold_dict['upper']) & \
                (verts[:, 0] >= threshold_dict['sleeve_right']) & \
                (verts[:, 0] <= threshold_dict['sleeve_left'])
                
        def _lower_mask(verts, threshold_dict):
            return verts[:, 1] >= threshold_dict['lower']
        
        for patch_label in self.full_patch_idxs_dict:
            patch_verts = self.active_mesh['vertices'][self.full_patch_idxs_dict[patch_label]]
            mask_fun = _upper_mask if patch_label.split('_')[0] == 'upper' else _lower_mask
            mask = mask_fun(patch_verts, self.threshold_dict)
            self.masked_patch_idxs_dict[patch_label] = self.full_patch_idxs_dict[patch_label][mask]
            
        # NOTE: Masked seamlines dict not needed at the moment, since old-to-new dict finally masks the seamlines as well
        for seamline_component in self.ref_seamlines_dict:
            seamline_verts = self.ref_seamlines_dict[seamline_component]
            mask_fun = _lower_mask if patch_label.split('_')[0] == 'lower' else _upper_mask
            mask = mask_fun(seamline_verts, self.threshold_dict)
            #self.masked_seamlines_dict[seamline_component] = seamline_verts[mask]
        
    def update_garment_meshes(self):
        self.apply_length_params()
        self.apply_masks()
        
    def update_parameter(self, param_name, param_value, update_meshes=True):
        ''' Only updates the parameter. '''
        self.design_params.update_parameter(param_name, param_value)
        if update_meshes:
            self.update_garment_meshes()
    
    def update_parameters(self, design_params):
        if type(design_params) == str:
            with open(f'config/designs/{design_params}.json', 'r') as json_file:
                design_params = json.load(json_file)
        if type(design_params) == dict:
            design_params = DesignParameters(design_dict=design_params)
        self.design_params = design_params
        self.update_garment_meshes()
        
    def finalize(self):
        self._extract_patch_meshes()
        self._update_seamlines()
        if self.use_darts:
            self._create_darts()
        
        if self.apply_remesh:
            self._apply_remesh()
        
        self._extract_uv_unit_directions()
        self._extract_reference_local_stretches()
        
        if self.use_darts:
            self._update_darts()
        
        self._export_prepared_data()    # TODO: Later, directly pass data as arguments to the C++ via PyBind
        
    def _get_active_template_smpls(self):
        active_template_smpl_dict = {}
        if self.ref_gender == 'female':
            active_template_smpl_dict[self.ref_gender] = SMPL(
                model_path=os.path.join(SMPL_DIR, f'SMPL_{self.ref_gender.upper()}.pkl'), 
                gender=self.ref_gender,
                v_template=torch.from_numpy(self.active_mesh['vertices']).float()
            )
        else:   # there seems to be a bug when using v_template with male models so more suitable not to use the cut mesh (especially for pants)
            active_template_smpl_dict[self.ref_gender] = SMPL(
                model_path=os.path.join(SMPL_DIR, f'SMPL_{self.ref_gender.upper()}.pkl'), 
                gender=self.ref_gender
            )
        trimesh.Trimesh(vertices=self.active_mesh['vertices'], faces=self.active_mesh['faces']).export('controversial_active_mesh.ply')
        if self.body_set.target_genders_differ_from_ref:
            raise Exception("Avoid using this for the moment...")
            remaining_gender = 'male' if self.ref_gender == 'female' else 'female'
            other_template_verts = MeshProcessor.update_homologous_mesh(
                target_gender=remaining_gender,
                barycentric_coords=self.bary_coords_for_active_cuts_dict
            )
            active_template_smpl_dict[remaining_gender] = SMPL(
                model_path=os.path.join(SMPL_DIR, f'SMPL_{remaining_gender.upper()}.pkl'), 
                gender=remaining_gender,
                v_template=torch.from_numpy(other_template_verts).float()
            )
        return active_template_smpl_dict
        
    def _get_target_posed_verts(self):
        active_template_smpl_dict = self._get_active_template_smpls()   # will typically contain only the reference gender
        target_posed_verts_list = []
        for body_idx in range(self.body_set.num_targets):
            target_pose_fun = getattr(const, self.body_set.target['poses'][body_idx])
            target_shape_fun = getattr(const, self.body_set.target['shapes'][body_idx])
            target_gender = self.body_set.target['genders'][body_idx]
            
            target_posed_verts_list.append(active_template_smpl_dict[target_gender](
                body_pose=target_pose_fun(), 
                betas=target_shape_fun()
            ).vertices[0].cpu().detach().numpy())
            trimesh.Trimesh(vertices=active_template_smpl_dict[target_gender](
                body_pose=target_pose_fun(), 
                betas=target_shape_fun()
            ).vertices[0].cpu().detach().numpy(), faces=self.active_mesh['faces']).export('controversial_mesh.ply')
        return target_posed_verts_list
    
    def _select_active_faces(self, patch_vert_idxs):
        return [face for face in self.active_mesh['faces'] if set(face).issubset(patch_vert_idxs)]
    
    @staticmethod
    def _extract_old_to_new_vert_idxs_map(patch_vert_idxs):
        return {old_index: new_index for new_index, old_index in enumerate(patch_vert_idxs)}
    
    def _map_old_to_new_faces(self, patch_label, patch_faces):
        new_face_list = []
        for face in patch_faces:
            new_face = []
            for v in face:
                new_face.append(self.old_to_new_idx_dict[patch_label][v])
            new_face_list.append(new_face)
        return np.array(new_face_list)
        
    def _extract_patch_meshes(self):
        target_posed_verts_list = self._get_target_posed_verts()
        
        trimesh.Trimesh(vertices=self.active_mesh['vertices'], faces=self.active_mesh['faces']).export('active_mesh.ply')
        #trimesh.Trimesh(vertices=target_posed_verts_list[0], faces=self.active_mesh['faces']).export('target_mesh.ply')
        
        for patch_label in PATCH_LIST:
            patch_vert_idxs = self.masked_patch_idxs_dict[patch_label]
            patch_faces = self._select_active_faces(patch_vert_idxs)
            self.old_to_new_idx_dict[patch_label] = self._extract_old_to_new_vert_idxs_map(patch_vert_idxs)
            
            # Extract reference mesh patches
            self.ref_patch_verts_dict[patch_label] = self.active_mesh['vertices'][patch_vert_idxs]
            self.ref_patch_faces_dict[patch_label] = self._map_old_to_new_faces(patch_label, patch_faces)
            
            trimesh.Trimesh(vertices=self.ref_patch_verts_dict[patch_label], faces=self.ref_patch_faces_dict[patch_label]).export(f'{patch_label}.ply')
            
            # Extract target mesh patches
            for body_idx in range(self.body_set.num_targets):
                self.target_bodies_verts.append(target_posed_verts_list[body_idx])
                self.target_patch_verts_dict_list[body_idx][patch_label] = target_posed_verts_list[body_idx][patch_vert_idxs]
                self.target_patch_faces_dict_list[body_idx][patch_label] = self.ref_patch_faces_dict[patch_label]
                
    def _update_seamlines(self):
        ''' Updates seamline vertex indices based on the selected garment patch vertices.
        
        Each patch has an ID. Each patch has multiple seamlines that they share with exactly
        one other patch. This structure is captured in `self.seam_to_segment_vertex_pairs`:
        
            - "<seam_label>": 
                - "<segment_id1>": [<list_of_seam_vert_idxs>]
                - "<segment_id2>": [<list_of_seam_vert_idxs>]
                
        The `map_old_to_new_indices` function selects only the vertex indices that are previously
        selected for that patch.
        '''
        def map_old_to_new_indices(patch_label, old_indices):
            new_indices = []
            for old_index in old_indices:
                # NOTE (crucial): Some old indices might not be in the old-to-new dictionary since they are boundary vertices below the garment length threshold
                if old_index in self.old_to_new_idx_dict[patch_label]:
                    new_indices.append(self.old_to_new_idx_dict[patch_label][old_index])
            return new_indices

        for patch_label in PATCH_LIST:
            patch_id = SEGMENT_TO_ID[patch_label]
            for seam_label in SEGMENT_TO_SEAMLINES_DICT[patch_id]:
                if seam_label not in self.seam_to_segment_vertex_pairs:
                    self.seam_to_segment_vertex_pairs[seam_label] = {
                        patch_id: map_old_to_new_indices(patch_label, SEAM_TO_SEAM_IDX_DICT[seam_label])
                    }
                else:
                    self.seam_to_segment_vertex_pairs[seam_label][patch_id] = \
                        map_old_to_new_indices(patch_label, SEAM_TO_SEAM_IDX_DICT[seam_label])
                        
    def _create_darts(self):
        for patch_label in PATCH_LIST:
            self.dart_dict[patch_label] = {}
            for dart_name in SEGMENT_TO_DARTS[patch_label]:
                dart_vertices = SEGMENT_TO_DARTS[patch_label][dart_name]
                new_darts_verts_idxs = [self.old_to_new_idx_dict[patch_label][v] for v in dart_vertices]
                self.ref_patch_verts_dict[patch_label], self.ref_patch_faces_dict[patch_label], dart_vertex_pairs = MeshProcessor.create_dart(
                    patch_verts=self.ref_patch_verts_dict[patch_label], 
                    patch_faces=self.ref_patch_faces_dict[patch_label], 
                    selected_vidxs=new_darts_verts_idxs,
                    dart_orient=DART_ORIENTS[dart_name]
                )
                self.dart_dict[patch_label][dart_name] = dart_vertex_pairs
            # TODO: Transfer darts to target meshes.
        
    def _get_seam_to_seam_endpoint_dict(self, patch_label):
        patch_id = SEGMENT_TO_ID[patch_label]
        seam_labels = SEGMENT_TO_SEAMLINES_DICT[SEGMENT_TO_ID[patch_label]]
        seamline_endpoints_dict = {}
        for seam_label in seam_labels:
            seamline_idxs = self.seam_to_segment_vertex_pairs[seam_label][patch_id]
            seamline_endpoints_dict[seam_label] = (seamline_idxs[0], seamline_idxs[-1])
        return seamline_endpoints_dict
        
    def _apply_remesh(self):
        for patch_label in PATCH_LIST:
            new_verts, new_faces, new_seamline_vert_idxs = apply_remesh(
                vertices=self.ref_patch_verts_dict[patch_label],
                faces=self.ref_patch_faces_dict[patch_label],
                seamline_pairs_dict=self._get_seam_to_seam_endpoint_dict(patch_label)
            )
            self.ref_patch_verts_dict[patch_label] = new_verts
            self.ref_patch_faces_dict[patch_label] = new_faces
            self.seam_to_segment_vertex_pairs[patch_label] = new_seamline_vert_idxs
            
            for body_idx in range(self.body_set.num_targets):
                transfer_remesh_to_target(
                    source_vertices=new_verts,
                    source_faces=new_faces,
                    target_vertices_list=self.target_patch_verts_dict_list[body_idx][patch_label]
                )
        
    def _extract_uv_unit_directions(self):
        for patch_label in PATCH_LIST:
            self.uv_unit_directions_dict[patch_label] = MeshProcessor.compute_triangle_directions(
                vertices=self.ref_patch_verts_dict[patch_label],
                faces=self.ref_patch_faces_dict[patch_label]
            )
    
    def _extract_reference_local_stretches(self):
        for patch_label in PATCH_LIST:
            self.local_stretches_dict[patch_label] = MeshProcessor.extract_local_stretches(
                verts=self.ref_patch_verts_dict[patch_label],
                faces=self.ref_patch_faces_dict[patch_label],
                design_params=self.design_params,
                patch_label=patch_label
            )
            
    def _update_darts(self):
        for patch_label in PATCH_LIST:
            for dart_name in self.dart_dict[patch_label]:
                dart_vert_pairs = self.dart_dict[patch_label][dart_name]
                dart_side = dart_name.split('_')[0]
                # The 'dart_side' indicator is used to swap (right, left) (for right darts) to (left, right) (for right darts).
                # This is important to properly traverse the edges but is a bit tedious, for now.
                # Most darts could be right darts, except for the left-armpit dart, where we need to swap.
                if dart_side == 'right':
                    _, inner_triangle_area = select_faces_in_dart(
                        vertices=self.ref_patch_verts_dict[patch_label], 
                        faces=self.ref_patch_faces_dict[patch_label], 
                        left_dart_cut_idx=dart_vert_pairs[-1][1],    # (right, left) dart vertex pairs
                        right_dart_cut_idx=dart_vert_pairs[-1][0], 
                        dart_tip_idx=dart_vert_pairs[0], 
                        dart_size=0.05,
                        max_distance_from_plane=0.1,
                        dart_side=dart_name.split('_')[0]
                    )
                    # NOTE: For now, not using the fractions, only the "full" faces
                    outer_face_fractions_dict, outer_triangle_area = select_faces_in_dart(
                        vertices=self.ref_patch_verts_dict[patch_label], 
                        faces=self.ref_patch_faces_dict[patch_label], 
                        left_dart_cut_idx=dart_vert_pairs[-1][1], 
                        right_dart_cut_idx=dart_vert_pairs[-1][0], 
                        dart_tip_idx=dart_vert_pairs[0], 
                        dart_size=0.1,
                        max_distance_from_plane=0.1,
                        dart_side=dart_name.split('_')[0]
                    )
                else:
                    _, inner_triangle_area = select_faces_in_dart(
                        vertices=self.ref_patch_verts_dict[patch_label], 
                        faces=self.ref_patch_faces_dict[patch_label], 
                        left_dart_cut_idx=dart_vert_pairs[-1][0],    # (left, right) dart vertex pairs
                        right_dart_cut_idx=dart_vert_pairs[-1][1], 
                        dart_tip_idx=dart_vert_pairs[0], 
                        dart_size=0.05,
                        max_distance_from_plane=0.1,
                        dart_side=dart_name.split('_')[0]
                    )
                    # NOTE: For now, not using the fractions, only the "full" faces
                    outer_face_fractions_dict, outer_triangle_area = select_faces_in_dart(
                        vertices=self.ref_patch_verts_dict[patch_label], 
                        faces=self.ref_patch_faces_dict[patch_label], 
                        left_dart_cut_idx=dart_vert_pairs[-1][0], 
                        right_dart_cut_idx=dart_vert_pairs[-1][1], 
                        dart_tip_idx=dart_vert_pairs[0], 
                        dart_size=0.1,
                        max_distance_from_plane=0.1,
                        dart_side=dart_name.split('_')[0]
                    )
                # NOTE: The new coefficient is simply a ratio between the difference between the larger and smaller triangle divided by the larger times original coefficient
                # TODO: Later, perhaps, use face fractions to more precisely update the coefficients
                outer_inner_area_ratio = ((outer_triangle_area - inner_triangle_area) / outer_triangle_area)
                for face_idx in outer_face_fractions_dict:
                    # TODO: Do not use fixed 'v' direction
                    # Instead, determine the direction based on 1) dart orientation, 2) sleeve/not-sleeve
                    self.local_stretches_dict[patch_label]['v'][face_idx] /= outer_inner_area_ratio
                    
        colored_upper_front_mesh = color_code_stretches(
            verts=self.ref_patch_verts_dict['upper_front'],
            faces=self.ref_patch_faces_dict['upper_front'],
            stretch_array=self.local_stretches_dict['upper_front']['v'],
            min_stretch=self.local_stretches_dict['upper_front']['v'].min(),
            max_stretch=self.local_stretches_dict['upper_front']['v'].max()
        )
        colored_upper_front_mesh.export('dart_stretches_debug.ply')
            
    def _export_seamline_vertex_pairs(self, seamline_dir):
        for seam_name in self.seam_to_segment_vertex_pairs:
            save_seamline_pairs_file(
                fpath=os.path.join(seamline_dir, f'{seam_name}.txt'),
                seamline_pair_dict=self.seam_to_segment_vertex_pairs[seam_name]
            )
        for patch_label in self.dart_dict:
            darts_dir = f'data/darts/{patch_label}'
            os.makedirs(os.path.join(darts_dir), exist_ok=True)
            save_darts_files(darts_dir, self.dart_dict[patch_label])
        
    @staticmethod
    def check_unique_with_tolerance(vertices, tolerance=1e-4):
        rounded = np.round(vertices / tolerance) * tolerance
        return len(np.unique(rounded, axis=0)) == len(vertices)
        
    def _export_prepared_data(self):
        mesh_dir = 'data/embedded/'
        body_dir = 'data/body/'
        seamline_dir = f'data/seamlines/'
        stretch_dir = 'data/scales/'
        uv_ref_3d_dir = 'data/bary/ref_3d/'
        uv_ref_2d_dir = 'data/bary/ref_2d/' # only create the director(ies) and fill out in C++
        for dir in [mesh_dir, body_dir, seamline_dir]:
            if os.path.exists(dir):
                shutil.rmtree(dir)
                os.makedirs(dir)
        
        trimesh.Trimesh(
            vertices=self.active_mesh['vertices'], faces=self.active_mesh['faces']).export(os.path.join(body_dir, 'ref.ply'))
        
        for patch_label in PATCH_LIST:
            # Store reference embedded garment meshes (mesh data)
            embedded_subdir = os.path.join(mesh_dir, patch_label)
            os.makedirs(embedded_subdir, exist_ok=True)
            export(
                verts=self.ref_patch_verts_dict[patch_label],
                faces=self.ref_patch_faces_dict[patch_label],
                path=os.path.join(embedded_subdir, f'ref')
            )
            print(f'Exporting {embedded_subdir}/ref.(ply/obj)...')
            
            # Store reference local-triangle design scales (local scalars) to txt
            stretch_subdir = os.path.join(stretch_dir, patch_label)
            os.makedirs(stretch_subdir, exist_ok=True)
            np.savetxt(
                os.path.join(stretch_subdir, f'scales_u.txt'), 
                self.local_stretches_dict[patch_label]['u']
            )
            np.savetxt(
                os.path.join(stretch_subdir, f'scales_v.txt'), 
                self.local_stretches_dict[patch_label]['v']
            )
            print(f'Exporting {stretch_subdir}/scales_(u/v).txt...')
            
            # Store reference UV directions (global axes projected to local triangles) in bary coordinates
            uv_ref_3d_subdir = os.path.join(uv_ref_3d_dir, patch_label)
            uv_ref_2d_subdir = os.path.join(uv_ref_2d_dir, patch_label)
            os.makedirs(uv_ref_3d_subdir, exist_ok=True)
            os.makedirs(uv_ref_2d_subdir, exist_ok=True)
            np.savetxt(
                os.path.join(uv_ref_3d_subdir, f'ref_bary_u.txt'),
                self.uv_unit_directions_dict[patch_label][:, 0])
            np.savetxt(
                os.path.join(uv_ref_3d_subdir, f'ref_bary_v.txt'),
                self.uv_unit_directions_dict[patch_label][:, 1]
            )
            print(f'Exporting {uv_ref_3d_subdir}/ref_bary_(u/v).txt...')
            
            # Store seamline vertex pairs
            self._export_seamline_vertex_pairs(seamline_dir)
            
            for body_idx in range(len(self.body_set.target['poses'])):
                trimesh.Trimesh(
                    vertices=self.target_bodies_verts[body_idx], faces=self.active_mesh['faces']).export(os.path.join(body_dir, f'target-{body_idx:2d}.ply'))
                trimesh.Trimesh(
                    vertices=self.target_bodies_verts[body_idx], faces=self.active_mesh['faces']).export(os.path.join(body_dir, f'target-{body_idx:2d}.obj'))
                export(
                    verts=self.target_patch_verts_dict_list[body_idx][patch_label],
                    faces=self.target_patch_faces_dict_list[body_idx][patch_label],
                    path=os.path.join(embedded_subdir, f'target-{body_idx:02d}')
                )
                print(f'Exporting {embedded_subdir}/target-{body_idx:02}.(ply/obj)...')
        
    def optimize(self):
        os.makedirs('data/param_2d/', exist_ok=True)
        for patch_label in PATCH_LIST:
            os.makedirs(f'data/param_2d/{patch_label}', exist_ok=True)
            
        run_parameterization(
            config=self.config
        )  # TODO: Use this through pybind
        
    def has_all_meshes(self):
        return all(mesh['vertices'] is not None and mesh['faces'] is not None 
                  for mesh in [self.body_mesh, self.upper_garment, self.lower_garment])


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


def load_smpl(smpl_dir, gender):
    return SMPL(model_path=os.path.join(smpl_dir, f'SMPL_{gender.upper()}.pkl'), gender=gender)


def load_canonical_mesh(smpl_dir, gender):
    smpl_model = SMPL(model_path=os.path.join(smpl_dir, f'SMPL_{gender.upper()}.pkl'), gender=gender)
    canonical_verts = smpl_model().vertices[0].cpu().detach().numpy()
    
    return canonical_verts, smpl_model.faces


def init_static(args):
    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)
    smpl_model = load_smpl(args.smpl_dir, args.gender)
    design_params = DesignParameters(design_dict)
    body_set = BodySet(set_dict)
    mesh_state = MeshState(smpl_model)
    
    return design_params, body_set, mesh_state


def init_garment_for_preselect(female_model):
    with open(f'config/designs/full.json', 'r') as json_file:
        design_dict = json.load(json_file)

    verts = female_model().vertices[0].cpu().detach().numpy()
    faces = female_model.faces
    garment = Garment(verts, faces, use_darts=False)

    return garment, design_dict


def initialize_garment_and_configs(args, female_model):
    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)

    verts = female_model().vertices[0].cpu().detach().numpy()
    faces = female_model.faces
    garment = Garment(verts, faces, use_darts=args.use_darts)

    return garment, design_dict, set_dict


def process_garment_set(args, modified_models, garment, set_dict, garment_parts, offset_type):
    for set_element_idx in range(len(set_dict['poses'])):
        pose_fun = getattr(const, set_dict['poses'][set_element_idx])
        shape_fun = getattr(const, set_dict['shapes'][set_element_idx])
        gender = set_dict['genders'][set_element_idx]
        
        posed_verts = modified_models[gender](
            body_pose=pose_fun(), 
            betas=shape_fun()
        ).vertices[0].cpu().detach().numpy()

        mesh_name = 'init' if set_element_idx == 0 else f'target-{set_element_idx:02d}'
        mesh_set_dir = os.path.join('data/embedded/', f'{args.design}-{args.body_set}/{offset_type}')
        latest_set_dir = os.path.join('data/embedded/latest/', offset_type)

        create_directories(mesh_set_dir, latest_set_dir)
        garment_parts = process_garment_parts(garment, garment_parts, posed_verts, offset_type)
        
        for part_name in garment_parts:
            part_verts = garment_parts[part_name]['verts']
            part_faces = garment_parts[part_name]['faces']
            export(args, part_verts, part_faces, f'{mesh_set_dir}/{part_name}/{mesh_name}', args.file_format)
            export(args, part_verts, part_faces, f'{latest_set_dir}/{part_name}/{mesh_name}', args.file_format)
            
        export_body_mesh(args, posed_verts, modified_models[gender].faces, set_element_idx, mesh_set_dir, latest_set_dir, gender)


def flood_fill_all_parts(garment, modified_verts, seams_info):
    garment_parts = {}
    
    # Upper garment (shirt) flood fill
    garment_parts['upper_front'] = garment.flood_fill_vertices(
        vertex_positions=modified_verts, 
        boundary_vertices=seams_info['upper_front'], 
        y_threshold=seams_info['y_upper_threshold'], 
        start_vertex=INIT_UPPER_FRONT
    )
    garment_parts['upper_back'] = garment.flood_fill_vertices(
        vertex_positions=modified_verts, 
        boundary_vertices=seams_info['upper_back'], 
        y_threshold=seams_info['y_upper_threshold'], 
        start_vertex=INIT_UPPER_BACK
    )

    # Sleeve flood fill
    sleeve_init_points = {
        'sleeve_front_right': INIT_FRONT_RIGHT_SLEEVE,
        'sleeve_back_right': INIT_BACK_RIGHT_SLEEVE,
        'sleeve_front_left': INIT_FRONT_LEFT_SLEEVE,
        'sleeve_back_left': INIT_BACK_LEFT_SLEEVE
    }

    for key, init_idx in sleeve_init_points.items():
        garment_parts[key] = garment.flood_fill_sleeve_vertices(
            vertex_positions=modified_verts, 
            boundary_vertices=seams_info[key]['seams'], 
            start_vertex=init_idx, 
            x_threshold=seams_info[key]['threshold'], 
            side=key.split('_')[-1]
        )

    # Lower garment (pant) flood fill
    lower_init_points = {
        'lower_front_right': INIT_RIGHT_FRONT_PANT,
        'lower_front_left': INIT_LEFT_FRONT_PANT,
        'lower_back_right': INIT_RIGHT_BACK_PANT,
        'lower_back_left': INIT_LEFT_BACK_PANT
    }

    for key, init_idx in lower_init_points.items():
        garment_parts[key] = garment.flood_fill_vertices(
            vertex_positions=modified_verts, 
            boundary_vertices=seams_info[key]['seams'],
            y_threshold=seams_info[key]['threshold_low'], 
            start_vertex=init_idx,
            up_pant_threshold=seams_info[key]['threshold_up']
        )

    return garment_parts


def process_garment_parts(garment, garment_parts, posed_verts, offset_type):
    garment_parts = {}
    for part_name in SEGMENT_SETS['default']:
        part_verts, part_faces = garment.extract_garment_mesh(
            posed_verts, 
            garment.mesh.faces, 
            garment_parts[part_name],
            offset=DISPLACEMENTS[offset_type], 
            segment_name=part_name
        )
        garment_parts[part_name] = {
            'verts': part_verts,
            'faces': part_faces
        }
    return garment_parts


def collect_vert_idxs_of_full_garments(patch_idxs_dict):
    patch_idxs_dict['upper'] = \
            patch_idxs_dict['upper_front'] + patch_idxs_dict['upper_back'] + \
            patch_idxs_dict['sleeve_front_right'] + patch_idxs_dict['sleeve_back_right'] + \
            patch_idxs_dict['sleeve_front_left'] + patch_idxs_dict['sleeve_back_left']
    patch_idxs_dict['lower'] = \
            sum([patch_idxs_dict[f'lower_{part}'] for part in ['front_right', 'front_left', 'back_right', 'back_left']], [])
    return patch_idxs_dict
    
    
def apply_length_params(mesh_state, garment_params):
    component_lengths_dict = {
        'upper': garment_params.shirt_length, 
        'sleeve': garment_params.sleeve_length,
        'lower': garment_params.pant_length
    }
    for component_label in ['upper', 'sleeve_left', 'sleeve_right', 'lower']:
        component_length = component_lengths_dict[component_label.split('_')[0]]
        mesh_state.threshold_dict[component_label] = FIXED_POINTS_DICT[component_label] + COMPONENT_SIGN_DICT[component_label] * component_length
        mesh_state.active_mesh['vertices'] = modify_mesh_with_plane_cut(
            vertices=mesh_state.canonical_mesh['vertices'],
            faces=mesh_state.canonical_mesh['faces'],
            cutting_point=mesh_state.threshold_dict[component_label],
            plane_orientation=PLANE_ORIENT_DICT[component_label],
            sleeve_side=component_label.split('_')[1] if component_label[:6] == 'sleeve' else None
        )
    return mesh_state


def flood_fill(self, vertex_positions, boundary_vertices, start_vertex, garment_label, threshold, **kwargs):
    """
    Unified flood fill function that automatically handles different garment types.
    
    Parameters:
    - vertex_positions: Array of vertex positions
    - boundary_vertices: List of boundary vertex indices
    - start_vertex: Starting vertex index for flood fill
    - garment_label: String indicating garment type ('upper', 'lower', or 'sleeve')
    - threshold: Main threshold value (interpreted based on garment_label)
    - **kwargs: Additional arguments:
        - upper_threshold: Optional upper bound for Y threshold (for upper/lower garments)
        - side: 'left' or 'right' for sleeves
    
    Returns:
    - List of selected vertex indices
    """
    def threshold_check(pos, idx):
        if garment_label == 'sleeve':
            side = kwargs.get('side', 'right')
            return pos[idx, 0] > threshold if side == 'right' else pos[idx, 0] < threshold
        else:
            # For upper and lower garments, use Y threshold
            condition = pos[idx, 1] >= threshold
            upper_threshold = kwargs.get('upper_threshold')
            if upper_threshold is not None:
                condition = condition and pos[idx, 1] <= upper_threshold
            return condition

    # Convert boundary vertices to a set for efficient lookup
    boundary_set = set(boundary_vertices)
    
    # Initialize stack, visited set, and selected vertices set
    stack = [start_vertex]
    visited = set()
    selected_vertices = set()
    
    # Main flood fill loop
    while stack:
        vertex_idx = stack.pop()
        
        # Check if vertex should be processed
        if (vertex_idx not in visited and 
            vertex_idx not in boundary_set and 
            threshold_check(vertex_positions, vertex_idx)):
            
            # Mark vertex as visited and selected
            visited.add(vertex_idx)
            selected_vertices.add(vertex_idx)
            
            # Add unvisited neighbors to stack
            for neighbor_idx in self.vertex_adjacency_list[vertex_idx]:
                if neighbor_idx not in visited:
                    stack.append(neighbor_idx)
    
    # Add qualifying boundary vertices
    thresh_boundaries = [idx for idx in boundary_vertices 
                        if threshold_check(vertex_positions, idx)]
    selected_vertices.update(thresh_boundaries)
    
    return list(selected_vertices)


def build_vertex_adjacency_list(F):
    adjacency_list = {}
    for triangle in F:
        for vertex in triangle:
            if vertex not in adjacency_list:
                adjacency_list[vertex] = set()
            # Add the neighboring vertices
            adjacency_list[vertex].update(triangle)
            adjacency_list[vertex].remove(vertex)  # A vertex is not a neighbor to itself
    return adjacency_list


def apply_pre_seams(mesh_state):
    vertex_adjacency_list = build_vertex_adjacency_list(mesh_state.canonical_mesh['faces'])
    patch_idxs_dict = {}
    for patch_label in INIT_IDXS:
        patch_type = patch_label.split('_')[0]
        patch_vert_idxs = flood_fill(
            vertex_adjacency_list=vertex_adjacency_list,
            vertex_positions=mesh_state.active_mesh['vertices'],
            boundary_vertices=PRE_SEAMS_DICT[patch_type],
            start_vertex=INIT_IDXS[patch_label],
            patch_type=patch_type,
            threshold=mesh_state.threshold_dict[patch_type],
            side=patch_label.split('_')[2] if patch_type == 'sleeve' else None
        )
        patch_idxs_dict[patch_label] = patch_vert_idxs
    return patch_idxs_dict


def apply_masks(patch_idxs_dict, modified_verts, threshold_dict):
    
    def _upper_mask(verts, threshold_dict):
        return (verts[:, 1] > threshold_dict['upper']) & \
               (verts[:, 0] > threshold_dict['sleeve_right']) & \
               (verts[:, 0] < threshold_dict['sleeve_left'])
               
    def _lower_mask(verts, threshold_dict):
        return verts[:, 1] > threshold_dict['lower']
    
    for patch_label in patch_idxs_dict:
        patch_verts = modified_verts[patch_idxs_dict[patch_label]]
        mask_fun = _upper_mask if patch_label.split('_')[0] == 'upper' else _lower_mask
        mask = mask_fun(patch_verts, threshold_dict)
        patch_verts[patch_label] = patch_verts[patch_label][mask]
        
    return patch_idxs_dict
    

def apply_all_cuts(seams_dict, smpl_model, upper_verts, lower_verts, modified_verts):
    # TODO: Apply other types of cuts (other seamlines, darts, and cuts).
    return seams_dict


def preselect(args, _unused_set_dict):
    # NOTE: Only the default (zero female) shape is now supported as initial.
    TEMPLATE_DIR_NAME = 'female-zero_shape'
    template_dir_path = f'templates/{TEMPLATE_DIR_NAME}/'
    if not os.path.exists(template_dir_path):
        print(f'(generating new template - {TEMPLATE_DIR_NAME}...)')
        os.makedirs(template_dir_path)
        init_design_params, _, mesh_state = init_static(args)
        mesh_state = apply_length_params(mesh_state, init_design_params)
        
        patch_idxs_dict = apply_pre_seams(mesh_state)
        patch_idxs_dict = collect_vert_idxs_of_full_garments(patch_idxs_dict)
        
        store_preselected(template_dir_path, patch_idxs_dict, mesh_state.active_mesh['vertices'])
    else:
        print(f'(pre-loading existing template - {TEMPLATE_DIR_NAME}...)')
        patch_idxs_dict, init_active_verts = load_preselected(template_dir_path)
        
    return patch_idxs_dict, init_active_verts


def load_garments(args):
    patch_idxs_dict, init_active_verts = preselect(args)   # NOTE: patch dict also includes whole garments
    return init_active_verts[patch_idxs_dict['upper']], init_active_verts[patch_idxs_dict['lower']]
   
    
def update_garment_meshes(garment_params):
    modified_verts, threshold_dict = apply_length_params(mesh_state, garment_params)
    patch_idxs_dict = apply_masks(patch_idxs_dict, modified_verts, threshold_dict)
