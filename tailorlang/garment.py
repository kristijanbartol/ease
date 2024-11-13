from typing import Dict, Tuple
import numpy as np
import trimesh
import os
from collections import defaultdict

from tailorlang.geometry import apply_offset_to_verts
from tailorlang.const import (
    SEGMENT_TO_SEAMLINES_DICT,
    SEGMENT_TO_ID,
    SEAM_TO_SEAM_IDX_DICT,
    SEGMENT_TO_DARTS,
    DART_ORIENTS
)
from tailorlang.utils import (
    save_seamline_pairs_file,
    save_darts_files
)


def point_side(p, edge, dart_orient):
    return dart_orient * np.cross(edge[1] - edge[0], p - edge[0])[1]  # Assuming Y is up


def map_vertex_idx_to_face_idxs(faces):
    # Create a map of vertex to faces for quick lookup
    vertex_idx_to_face_idxs = defaultdict(list)
    for i, face in enumerate(faces):
        for v in face:
            vertex_idx_to_face_idxs[v].append(i)
    return vertex_idx_to_face_idxs


def get_adjacent_faces(face1, face2):
    return set(face1) & set(face2)


def update_face(face, v_idx_to_update, new_v_idx):
    # To update the vertex in the face, iterate through the triangle.
    for idx, v_idx in enumerate(face):
        if v_idx == v_idx_to_update:
            face[idx] = new_v_idx
    return face


def create_dart(vertices, faces, selected_vidxs, dart_orient):
    # New vertex idx map contains the map (orig_selected_idx->new_vertex_idx).
    new_vertex_idx_map = {v_idx: len(vertices) + idx for idx, v_idx in enumerate(selected_vidxs[:-1])}
    # Fast lookup of the corresponding face idxs, given vertex idx.
    vertex_idx_to_face_idxs = map_vertex_idx_to_face_idxs(faces)
    updated_faces = faces.copy()
    
    # Do not process the last edge, containing the dart tip.
    for idx, _ in enumerate(selected_vidxs[:-2]):
        v1_idx, v2_idx = selected_vidxs[idx], selected_vidxs[idx+1]
        edge = vertices[v1_idx], vertices[v2_idx]
        adjacent_face_idxs = [f for f in vertex_idx_to_face_idxs[v1_idx]]
        
        for face_idx in adjacent_face_idxs:
            face = faces[face_idx]
            center = vertices[face].mean(axis=0)
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
    new_vertices = vertices[selected_vidxs[:-1]]
    new_vertices[:, 1] += 0.00001   # add tiny displacement so that trimesh doesn't skip "duplicate" vertices
    updated_vertices = np.vstack((vertices, new_vertices))
    
    # Create a dart vertex pair strategy relevant for further processing (parameterization).
    dart_pairs = [selected_vidxs[-1]]
    for i in range(len(selected_vidxs) - 2, -1, -1):
        dart_pairs.append((selected_vidxs[i], new_vertex_idx_map[selected_vidxs[i]]))

    return updated_vertices, updated_faces, dart_pairs


class Garment:
    def __init__(self, verts, faces, use_darts=False):
        self.mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        self.vertex_adjacency_list = self.build_vertex_adjacency_list(faces)
        self.seam_to_segment_vertex_pairs = {}
        self.segment_to_id = SEGMENT_TO_ID
        self.segment_to_seamlines_dict = SEGMENT_TO_SEAMLINES_DICT
        self.seam_to_seam_idx_dict = SEAM_TO_SEAM_IDX_DICT
        self.segment_to_edges_map = {}
        self.use_darts = use_darts
        self.dart_dict = {}

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

    def flood_fill_vertices(self, vertex_positions, boundary_vertices, y_threshold, start_vertex, up_pant_threshold=None):
        # Convert boundary vertices to a set for efficient lookup
        boundary_set = set(boundary_vertices)

        # Initialize the stack with the start vertex, which is assumed to be inside the boundary and above the y_threshold
        stack = [start_vertex]

        # Initialize the set to keep track of visited vertices
        visited = set()

        # Initialize the set to store the selected vertices
        selected_vertices = set()

        iter = 0
        while stack:
            # Pop the last vertex from the stack
            vertex_idx = stack.pop()

            # If the vertex has not been visited yet and is not on the boundary or below y_threshold, process it
            if up_pant_threshold is None:
                condition = vertex_idx not in visited and vertex_idx not in boundary_set and vertex_positions[vertex_idx, 1] >= y_threshold
            else:
                condition = vertex_idx not in visited and vertex_idx not in boundary_set and vertex_positions[vertex_idx, 1] >= y_threshold and vertex_positions[vertex_idx, 1] <= up_pant_threshold
            if condition:
                # Mark the vertex as visited and add to selected
                visited.add(vertex_idx)
                selected_vertices.add(vertex_idx)

                # Iterate over the neighbors of the current vertex
                for neighbor_idx in self.vertex_adjacency_list[vertex_idx]:
                    # If the neighbor hasn't been visited, add it to the stack
                    if neighbor_idx not in visited:
                        stack.append(neighbor_idx)
        
        # Once all possible vertices have been visited, combine the selected vertices with the boundary vertices
        # Note that only those boundary vertices whose Y coordinate is larger than a threshold should be selected
        thresh_boundaries = [index for index in boundary_vertices if vertex_positions[index][1] >= y_threshold]
        selected_vertices.update(thresh_boundaries)

        return list(selected_vertices)
    
    def flood_fill_sleeve_vertices(self, vertex_positions, boundary_vertices, start_vertex, x_threshold, side):

        def threshold_check(vertex_positions, vertex_idx, x_threshold, side):
            if side == 'right':
                return vertex_positions[vertex_idx, 0] > x_threshold
            else:
                return vertex_positions[vertex_idx, 0] < x_threshold
            
        sign = -1 if side == 'left' else 1

        # Convert boundary vertices to a set for efficient lookup
        boundary_set = set(boundary_vertices)

        # Initialize the stack with the start vertex, which is assumed to be inside the boundary and above the y_threshold
        stack = [start_vertex]

        # Initialize the set to keep track of visited vertices
        visited = set()

        # Initialize the set to store the selected vertices
        selected_vertices = set()

        while stack:
            # Pop the last vertex from the stack
            vertex_idx = stack.pop()

            # If the vertex has not been visited yet and is not on the boundary or below y_threshold, process it
            if vertex_idx not in visited and vertex_idx not in boundary_set and threshold_check(vertex_positions, vertex_idx, x_threshold, side):
                # Mark the vertex as visited and add to selected
                visited.add(vertex_idx)
                selected_vertices.add(vertex_idx)

                # Iterate over the neighbors of the current vertex
                for neighbor_idx in self.vertex_adjacency_list[vertex_idx]:
                    # If the neighbor hasn't been visited, add it to the stack
                    if neighbor_idx not in visited:
                        stack.append(neighbor_idx)
        
        # Once all possible vertices have been visited, combine the selected vertices with the boundary vertices
        # Note that only those boundary vertices whose Y coordinate is larger than a threshold should be selected
        thresh_boundaries = [index for index in boundary_vertices if threshold_check(vertex_positions, index, x_threshold, side)]
        selected_vertices.update(thresh_boundaries)

        return list(selected_vertices)

    def flood_fill_vertices_simplified(self, boundary_vertices, start_vertex):
        boundary_set = set(boundary_vertices)
        stack = [start_vertex]
        visited = set()
        selected_vertices = set()

        while stack:
            vertex_idx = stack.pop()
            if vertex_idx not in visited and vertex_idx not in boundary_set:
                visited.add(vertex_idx)
                selected_vertices.add(vertex_idx)

                for neighbor_idx in self.vertex_adjacency_list[vertex_idx]:
                    if neighbor_idx not in visited:
                        stack.append(neighbor_idx)
        
        selected_vertices.update(boundary_set)
        return list(selected_vertices)
    
    def update_seamline_vertex_pairs(self, old_to_new_index_mapping, segment_name):

        def map_old_to_new_indices(old_indices):
            new_indices = []
            for old_index in old_indices:
                # NOTE: Some old indices might not be in the old-to-new dictionary since they are boundary vertices below the garment length threshold
                if old_index in old_to_new_index_mapping:
                    new_indices.append(old_to_new_index_mapping[old_index])
            return new_indices

        segment_id = self.segment_to_id[segment_name]
        seamlines_list = self.segment_to_seamlines_dict[segment_id]
        for seam_name in seamlines_list:
            # The seam-to-segment-vertex-pairs dict is of structure:
            # { seam_name: {
            #     segment_id_1: new_verts1,
            #     segment_id_2: new_verts2   
            # }}
            if seam_name not in self.seam_to_segment_vertex_pairs:
                self.seam_to_segment_vertex_pairs[seam_name] = {
                    segment_id: map_old_to_new_indices(self.seam_to_seam_idx_dict[seam_name])
                }
            else:
                self.seam_to_segment_vertex_pairs[seam_name][segment_id] = \
                    map_old_to_new_indices(self.seam_to_seam_idx_dict[seam_name])
                    
    def process_darts_for_segment(self, segment_name, old_to_new_index_mapping, garment_verts, garment_faces):
        self.dart_dict[segment_name] = {}
        for dart_name in SEGMENT_TO_DARTS[segment_name]:
            dart_vertices = SEGMENT_TO_DARTS[segment_name][dart_name]
            new_darts_verts_idxs = [old_to_new_index_mapping[v] for v in dart_vertices]
            garment_verts, garment_faces, dart_vertex_pairs = create_dart(
                garment_verts, 
                garment_faces, 
                new_darts_verts_idxs,
                dart_orient=DART_ORIENTS[dart_name]
            )
            self.dart_dict[segment_name][dart_name] = dart_vertex_pairs
        return garment_verts, garment_faces
    
    def update_edge_mapping(self, segment_name, faces, vertex_mapping):
        """
        Create a mapping between old and new edge vertex indices.
        
        Parameters:
        faces (np.ndarray): Array of face vertex indices (N x 3)
        vertex_mapping (dict): Dictionary mapping old vertex indices to new vertex indices
        
        Result:
        self.segment_to_edges_map[segment_name]: Dictionary mapping old edge tuples to new edge tuples
            Format: {segment_name: {(old_v1, old_v2): (new_v1, new_v2)}}
        """
        # Create a set to store unique edges using new vertex indices
        edges = set()
        
        # Extract edges from faces
        for face in faces:
            # Get three edges from triangular face
            # Always store edges with smaller vertex index first for consistency
            v1, v2, v3 = sorted(face)
            edges.add((v1, v2))
            edges.add((v2, v3))
            edges.add((v1, v3))
        
        # Create reverse mapping (new to old vertex indices)
        reverse_mapping = {new: old for old, new in vertex_mapping.items()}
        
        # Create edge mapping dictionary
        edge_mapping = {}
        for new_v1, new_v2 in edges:
            # Get corresponding old vertex indices
            old_v1 = reverse_mapping[new_v1]
            old_v2 = reverse_mapping[new_v2]
            
            # Store with consistent ordering (smaller index first)
            old_edge = tuple(sorted([old_v1, old_v2]))
            new_edge = (new_v1, new_v2)  # Already sorted from earlier
            
            edge_mapping[old_edge] = new_edge
        
        self.segment_to_edges_map[segment_name] = edge_mapping
        
    def extract_garment_mesh(self, verts, faces, garment_vertex_indices, offset=0., segment_name=None):
        # Apply offset if specified
        if offset != 0:
            verts = apply_offset_to_verts(verts, faces, offset)

        # Extract vertices and faces for the garment
        garment_verts = verts[garment_vertex_indices]
        garment_faces = [face for face in faces if set(face).issubset(garment_vertex_indices)]

        # Create a mapping from the original vertex indices to the new, local indices
        old_to_new_index_mapping = {old_index: new_index for new_index, old_index in enumerate(garment_vertex_indices)}
        # Update the indices in the faces to the new local indices
        garment_faces = np.array([[old_to_new_index_mapping[v] for v in face] for face in garment_faces])

        if segment_name is not None:
            self.update_edge_mapping(
                segment_name=segment_name,
                faces=garment_faces,
                vertex_mapping=old_to_new_index_mapping
            )
            if self.use_darts:
                garment_verts, garment_faces = self.process_darts_for_segment(
                    segment_name, 
                    old_to_new_index_mapping, 
                    garment_verts, 
                    garment_faces
                )
            self.update_seamline_vertex_pairs(old_to_new_index_mapping, segment_name)

        return garment_verts, garment_faces
    
    def measure_optimized_edge_lengths(self, segment_to_mesh_dict: Dict[str, trimesh.Trimesh]) -> Dict[Tuple[int, int], float]:
        global_edge_dict = {}
        for segment_name in segment_to_mesh_dict:
            mesh = segment_to_mesh_dict[segment_name]
            for global_edge_pair in self.segment_to_edges_map[segment_name]:
                if global_edge_pair in global_edge_dict:
                    continue
                patch_verts = mesh.vertices
                local_edge_pair = self.segment_to_edges_map[segment_name][global_edge_pair]
                
                v1, v2 = patch_verts[local_edge_pair[0]], patch_verts[local_edge_pair[1]]
                edge_length = np.linalg.norm(v1 - v2)
                
                global_edge_dict[global_edge_pair] = edge_length
        return global_edge_dict
    
    def store_seamline_keypoints(self, subdir):
        pass
    
    def store_seamline_vertex_pairs(self, subdir):
        data_dir = 'data/seamlines/'
        active_subdir_path = os.path.join(data_dir, subdir)
        latest_subdir_path = os.path.join(data_dir, 'latest')
        os.makedirs(os.path.join(latest_subdir_path, 'debug'), exist_ok=True)
        for seam_name in self.seam_to_segment_vertex_pairs:
            save_seamline_pairs_file(
                fpath=os.path.join(active_subdir_path, f'{seam_name}.txt'),
                seamline_pair_dict=self.seam_to_segment_vertex_pairs[seam_name]
            )
            save_seamline_pairs_file(
                fpath=os.path.join(latest_subdir_path, f'{seam_name}.txt'),
                seamline_pair_dict=self.seam_to_segment_vertex_pairs[seam_name]
            )
        if self.use_darts:
            for segment_name in self.dart_dict:
                darts_dir = f'data/darts/latest/{segment_name}/'
                os.makedirs(os.path.join(darts_dir), exist_ok=True)
                save_darts_files(darts_dir, self.dart_dict[segment_name])
        
    @staticmethod
    def extract_garment_verts(verts, faces, garment_vertex_indices, offset=0.):
        if offset != 0:
            verts = apply_offset_to_verts(verts, faces, offset)

        return verts[garment_vertex_indices]
    