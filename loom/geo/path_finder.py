# NOTE: deprecated module, will remove soon...

import os
import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict
from smplx import SMPL

from loom.const import (
    LEFT_ARMPIT2,
    RIGHT_ARMPIT2,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_OUTER_PANT,
    RIGHT_OUTER_PANT,
    BACK_INNER_PANT,
    FRONT_INNER_PANT
)

class MeshPathFinder:
    def __init__(self, vertices, faces):
        """
        Initialize the mesh path finder with vertex positions and face information.
        
        Args:
            vertices: numpy array of shape (N, 3) containing vertex coordinates
            faces: numpy array of shape (M, 3) containing face vertex indices
        """
        self.vertices = vertices
        self.faces = faces
        self.adjacency = self._build_adjacency_list()
        
    def _build_adjacency_list(self):
        """Build vertex adjacency list from face information."""
        adjacency = defaultdict(set)
        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                adjacency[v1].add(v2)
                adjacency[v2].add(v1)
        return dict(adjacency)
    
    def _compute_angle(self, p1, p2, p3):
        """
        Compute angle between three points in 3D space.
        Returns angle in degrees.
        """
        v1 = self.vertices[p1] - self.vertices[p2]
        v2 = self.vertices[p3] - self.vertices[p2]
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Handle numerical precision issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def _find_next_vertex_angle_based(self, current, prev, target):
        """
        Find next vertex based on angle criteria.
        Selects the neighbor that creates an angle closest to 180 degrees.
        """
        neighbors = self.adjacency[current]
        best_angle_diff = float('inf')
        best_next = None
        
        for neighbor in neighbors:
            if neighbor == prev:
                continue
                
            angle = self._compute_angle(prev, current, neighbor)
            angle_diff = abs(180 - angle)
            
            if angle_diff < best_angle_diff:
                best_angle_diff = angle_diff
                best_next = neighbor
                
        return best_next
    
    def _compute_geodesic_distance(self, start, end):
        """
        Compute approximate geodesic distance between two vertices.
        Currently uses Euclidean distance as a simple approximation.
        """
        return np.linalg.norm(self.vertices[start] - self.vertices[end])
    
    def _find_next_vertex_geodesic(self, current, prev, target, angle_threshold=45):
        """
        Find next vertex using geodesic heuristic.
        Considers both distance to target and angle constraints.
        """
        neighbors = self.adjacency[current]
        target_dir = self.vertices[target] - self.vertices[current]
        target_dir /= np.linalg.norm(target_dir)
        
        best_score = float('inf')
        best_next = None
        
        for neighbor in neighbors:
            if neighbor == prev:
                continue
                
            # Check angle constraint
            if prev is not None:
                angle = self._compute_angle(prev, current, neighbor)
                if abs(180 - angle) > angle_threshold:
                    continue
            
            # Compute direction to neighbor
            neighbor_dir = self.vertices[neighbor] - self.vertices[current]
            neighbor_dir /= np.linalg.norm(neighbor_dir)
            
            # Score based on alignment with target direction
            alignment_score = 1 - np.dot(neighbor_dir, target_dir)
            
            if alignment_score < best_score:
                best_score = alignment_score
                best_next = neighbor
                
        return best_next
    
    def find_path(self, control_points):
        """
        Find path through mesh connecting multiple control points.
        
        Args:
            control_points: List of vertex indices defining control points
        
        Returns:
            List of vertex indices defining the path
        """
        if len(control_points) < 2:
            return control_points
            
        complete_path = [control_points[0]]
        
        # Process each pair of control points
        for i in range(len(control_points) - 1):
            start = control_points[i]
            end = control_points[i + 1]
            
            current = start
            prev = complete_path[-2] if len(complete_path) > 1 else None
            
            # Build path between current pair of control points
            while current != end:
                next_vertex = self._find_next_vertex_geodesic(current, prev, end)
                
                if next_vertex is None or next_vertex == current:
                    break  # Path cannot be found
                    
                prev = current
                current = next_vertex
                complete_path.append(current)
        
        return complete_path
    
    
if __name__ == '__main__':
    smpl_model = SMPL(model_path=os.path.join('/Users/kristijanbartol/data/smpl/models/', f'SMPL_FEMALE.pkl'), gender='female')
    verts = smpl_model().vertices[0].cpu().detach().numpy()
    faces = smpl_model.faces
    
    path_finder = MeshPathFinder(verts, faces)

    PREDEFINED_PATH_LISTS = [LEFT_ARMPIT2, RIGHT_ARMPIT2, LEFT_SHOULDER, RIGHT_SHOULDER, 
                             LEFT_OUTER_PANT, RIGHT_OUTER_PANT, BACK_INNER_PANT, FRONT_INNER_PANT]

    for path_list in PREDEFINED_PATH_LISTS:
        control_points_pair = (path_list[0], path_list[-1])
        geodesic_path = path_finder.find_path(control_points_pair)
        print(geodesic_path)
        print(path_list)
