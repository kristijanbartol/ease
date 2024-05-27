import numpy as np
import trimesh

from geometry import (
    apply_offset_to_verts,
    find_init_vertex_idx
)


class Garment:
    def __init__(self, verts, faces):
        self.mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        self.vertex_adjacency_list = self.build_vertex_adjacency_list(faces)
        self.face_adjacency_list = self.build_face_adjacency_list(faces)

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
    
    def flood_fill_vertices_subdivided(self, boundary_vertex_ids, starting_vertex_id):
        # Convert boundary vertices to a set for efficient lookup
        boundary_set = set(boundary_vertex_ids)

        # Initialize the stack with the start vertex, which is assumed to be inside the boundary and above the y_threshold
        stack = [starting_vertex_id]

        # Initialize the set to keep track of visited vertices
        visited = set()

        # Initialize the set to store the selected vertices
        selected_vertices = set()

        while stack:
            # Pop the last vertex from the stack
            vertex_idx = stack.pop()

            # If the vertex has not been visited yet and is not on the boundary or below y_threshold, process it
            if vertex_idx not in visited and vertex_idx not in boundary_set:
                # Mark the vertex as visited and add to selected
                visited.add(vertex_idx)
                selected_vertices.add(vertex_idx)

                # Iterate over the neighbors of the current vertex
                for neighbor_idx in self.vertex_adjacency_list[vertex_idx]:
                    # If the neighbor hasn't been visited, add it to the stack
                    if neighbor_idx not in visited:
                        stack.append(neighbor_idx)
        
        # Once all possible vertices have been visited, combine the selected vertices with the boundary vertices
        selected_vertices.update(boundary_vertex_ids)

        return list(selected_vertices)
    
    '''
    def select_faces(self, boundary_points, boundary_faces):
        """Select the inner faces using the Flood Fill algorithm."""
        starting_face = find_init_face(
            mesh=self.mesh,
            start_point=boundary_points.mean(axis=0)
        )
        stack = [starting_face]
        visited = set()
        selected = set()
        while stack:
            face = stack.pop()
            if face not in visited and face not in boundary_faces:
                visited.add(face)
                selected.add(face)
                for neighbor_face in self.face_adjacency_list[face]:
                    if neighbor_face not in visited:
                        stack.append(neighbor_face)
        selected.update(boundary_faces)
        return selected
    '''

    def select_faces(self, boundary_vertex_ids, boundary_points):
        starting_vertex_id = find_init_vertex_idx(
            mesh=self.mesh,
            start_point=boundary_points.mean(axis=0)
        )
        selected_verts = self.flood_fill_vertices_subdivided(
            boundary_vertex_ids=boundary_vertex_ids,
            starting_vertex_id=starting_vertex_id
        )
        selected_faces = []
        for face_idx, face in enumerate(self.mesh.faces):
            if face[0] in selected_verts and face[1] in selected_verts and face[2] in selected_verts:
                selected_faces.append(face_idx)
        return set(selected_faces), selected_verts

    def select_sleeve_verts(self, verts, start_vertex_index, seam_indices, sleeve_length, x_direction_multiplier):
        # Get the X coordinates of the starting and ending seam vertices
        x_start = verts[seam_indices[0], 0]

        # Define the range of X values based on the sleeve length and the direction multiplier
        if x_direction_multiplier == -1:
            # For the right sleeve, we find vertices backwards from the start
            x_min = x_start - sleeve_length
            x_max = x_start
        else:
            # For the left sleeve, we find vertices forwards from the start
            x_min = x_start
            x_max = x_start + sleeve_length

        # Ensure x_min is less than x_max
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)

        # Initialize the set for selected vertex indices
        selected_indices = set()
        
        # List to keep track of the vertices that need to be visited
        to_visit = [start_vertex_index]
        
        while to_visit:
            current_index = to_visit.pop()
            if current_index not in selected_indices and x_min <= verts[current_index, 0] <= x_max:
                selected_indices.add(current_index)
                # Get the indices of the vertices connected to the current vertex
                connected_vertices = self.vertex_adjacency_list[current_index]
                # Add unvisited vertices to the to_visit list
                for vertex_index in connected_vertices:
                    if vertex_index not in selected_indices:
                        to_visit.append(vertex_index)
        
        return list(selected_indices)

    @staticmethod
    def extract_garment_mesh(verts, faces, garment_vertex_indices, offset=0.):
        # Apply offset if specified
        if offset != 0:
            verts = apply_offset_to_verts(verts, faces, offset)

        # Extract vertices and faces for the garment
        garment_verts = verts[garment_vertex_indices]
        garment_faces = [face for face in faces if set(face).issubset(garment_vertex_indices)]

        # Create a mapping from the original vertex indices to the new, local indices
        index_mapping = {old_index: new_index for new_index, old_index in enumerate(garment_vertex_indices)}
        # Update the indices in the faces to the new local indices
        garment_faces = np.array([[index_mapping[v] for v in face] for face in garment_faces])

        return garment_verts, garment_faces
