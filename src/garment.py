import numpy as np
import trimesh
import os

from src.geometry import apply_offset_to_verts
from src.const import (
    SEGMENT_TO_SEAMLINES_DICT,
    SEGMENT_TO_ID,
    SEAM_TO_SEAM_IDX_DICT
)
from src.utils import save_seamline_pairs_file


class Garment:
    def __init__(self, verts, faces):
        self.mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        self.vertex_adjacency_list = self.build_vertex_adjacency_list(faces)
        self.face_adjacency_list = self.build_face_adjacency_list(faces)
        self.seam_to_segment_vertex_pairs = {}

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
    def build_face_adjacency_list(faces):
        # Initialize the adjacency list
        adjacency_list = {i: set() for i in range(len(faces))}
        
        # Initialize a vertex-to-face map
        vertex_to_face = {}
        for i, face in enumerate(faces):
            for vertex in face:
                if vertex not in vertex_to_face:
                    vertex_to_face[vertex] = set()
                vertex_to_face[vertex].add(i)
        
        # Build the adjacency list
        for face_index, face in enumerate(faces):
            for vertex in face:
                adjacent_faces = vertex_to_face[vertex]
                for adj_face_index in adjacent_faces:
                    if adj_face_index != face_index:
                        adjacency_list[face_index].add(adj_face_index)
        
        return adjacency_list

    def flood_fill_vertices(self, vertex_positions, boundary_vertices, y_threshold, start_vertex):
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
            if vertex_idx not in visited and vertex_idx not in boundary_set and vertex_positions[vertex_idx, 1] >= y_threshold:
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

        segment_id = SEGMENT_TO_ID[segment_name]
        seamlines_list = SEGMENT_TO_SEAMLINES_DICT[segment_id]
        for seam_name in seamlines_list:
            if seam_name not in self.seam_to_segment_vertex_pairs:
                self.seam_to_segment_vertex_pairs[seam_name] = {
                    segment_id: map_old_to_new_indices(SEAM_TO_SEAM_IDX_DICT[seam_name])
                }
            else:
                self.seam_to_segment_vertex_pairs[seam_name][segment_id] = \
                    map_old_to_new_indices(SEAM_TO_SEAM_IDX_DICT[seam_name])
    
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
            self.update_seamline_vertex_pairs(old_to_new_index_mapping, segment_name)

        return garment_verts, garment_faces
    
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
        
    @staticmethod
    def extract_garment_verts(verts, faces, garment_vertex_indices, offset=0.):
        if offset != 0:
            verts = apply_offset_to_verts(verts, faces, offset)

        return verts[garment_vertex_indices]
    