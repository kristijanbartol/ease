import numpy as np
import trimesh
import potpourri3d as pp3d
from scipy.spatial import KDTree
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ControlPoints:
    start: np.ndarray
    control: np.ndarray
    end: np.ndarray

class MeshSeamSmoother:
    def __init__(self, mesh: trimesh.Trimesh, num_samples: int = 10):
        self.mesh = mesh
        self.num_samples = num_samples
        self.kdtree = KDTree(self.mesh.vertices)
        # Initialize heat method solver
        self.path_solver = pp3d.EdgeFlipGeodesicSolver(
            self.mesh.vertices, 
            self.mesh.faces,
        )
        
    def _find_nearest_vertex_index(self, point: np.ndarray) -> int:
        """Find the index of the mesh vertex closest to the given point."""
        _, idx = self.kdtree.query(point)
        return idx
    
    def _compute_geodesic_path(self, start_point: np.ndarray, end_point: np.ndarray) -> np.ndarray:
        """Compute geodesic path between two points using heat method."""
        start_idx = self._find_nearest_vertex_index(start_point)
        end_idx = self._find_nearest_vertex_index(end_point)
        
        path = self.path_solver.find_geodesic_path(
            start_idx,
            end_idx
        )
        return np.array(path)
    
    def _geodesic_interpolate(self, p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
        """Interpolate between two points along the geodesic path."""
        path = self._compute_geodesic_path(p1, p2)
        
        # Compute cumulative distances along path
        segments = path[1:] - path[:-1]
        segment_lengths = np.linalg.norm(segments, axis=1)
        cumulative_lengths = np.cumsum(segment_lengths)
        total_length = cumulative_lengths[-1]
        
        # Find target distance
        target_distance = t * total_length
        
        # Find segment containing target distance
        if t <= 0:
            return path[0]
        elif t >= 1:
            return path[-1]
            
        segment_idx = np.searchsorted(cumulative_lengths, target_distance)
        if segment_idx == 0:
            prev_distance = 0
        else:
            prev_distance = cumulative_lengths[segment_idx - 1]
            
        # Interpolate within segment
        segment_t = (target_distance - prev_distance) / segment_lengths[segment_idx]
        point = path[segment_idx] + segment_t * segments[segment_idx]
        
        return point
    
    def _project_point_to_surface(self, point: np.ndarray) -> np.ndarray:
        """Project a point onto the nearest mesh face."""
        closest_point, _, face_index = self.mesh.nearest.on_surface([point])
        closest_point = closest_point[0]
        face_index = int(face_index[0])
        face_vertices = self.mesh.vertices[self.mesh.faces[face_index]]
        
        # Compute barycentric coordinates
        v0, v1, v2 = face_vertices
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = point - v0
        
        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1)
        d21 = np.dot(v0p, v0v2)
        
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        # Return interpolated point on face
        return u * v0 + v * v1 + w * v2
    
    def _project_points_to_surface(self, points: np.ndarray) -> np.ndarray:
        """Project points onto their nearest mesh faces.
        
        Args:
            points: Array of shape (N, 3) containing N 3D points
        Returns:
            Array of shape (N, 3) containing projected points
        """
        closest_points, _, face_indices = self.mesh.nearest.on_surface(points)
        face_indices = face_indices.astype(int)
        
        # Get vertices for each face
        faces = self.mesh.faces[face_indices]
        face_vertices = self.mesh.vertices[faces]  # Shape (N, 3, 3)
        
        # Compute barycentric coordinates for all points
        v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = points - v0
        
        d00 = np.sum(v0v1 * v0v1, axis=1)
        d01 = np.sum(v0v1 * v0v2, axis=1)
        d11 = np.sum(v0v2 * v0v2, axis=1)
        d20 = np.sum(v0p * v0v1, axis=1)
        d21 = np.sum(v0p * v0v2, axis=1)
        
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        # Return interpolated points on faces
        return u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2
    
    def _compute_bezier_point(self, control_points: ControlPoints, t: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute point on Bézier curve using adapted De Casteljau algorithm."""
        # First level interpolation
        q0 = self._geodesic_interpolate(control_points.start, control_points.control, t)
        q1 = self._geodesic_interpolate(control_points.control, control_points.end, t)
        
        # Second level interpolation
        b = self._geodesic_interpolate(q0, q1, t)
        
        #b = self._compute_geodesic_path(control_points.start, control_points.end)
        
        # Project point onto surface
        projected = self._project_point_to_surface(b)
        #projected = self._project_points_to_surface(b)
        return b, projected
    
    def _insert_vertex(self, point: np.ndarray) -> int:
        """Insert a new vertex into the mesh and update connectivity."""
        # Find closest face
        closest_point, _, face_index = self.mesh.nearest.on_surface([point])
        face_index = int(face_index[0])
        
        # Add new vertex
        new_vertex_idx = len(self.mesh.vertices)
        self.mesh.vertices = np.vstack((self.mesh.vertices, point))
        
        # Split the face into three new faces
        face_to_split = self.mesh.faces[face_index]
        new_faces = np.array([
            [face_to_split[0], face_to_split[1], new_vertex_idx],
            [face_to_split[1], face_to_split[2], new_vertex_idx],
            [face_to_split[2], face_to_split[0], new_vertex_idx]
        ])
        
        # Update faces
        self.mesh.faces = np.vstack((
            np.delete(self.mesh.faces, face_index, axis=0),
            new_faces
        ))
        
        return new_vertex_idx
    
    def save_points_as_ply(self, points: List[np.ndarray], output_path: str):
        """Save points as PLY point cloud."""
        points_array = np.array(points)
        point_cloud = trimesh.PointCloud(points_array)
        point_cloud.export(output_path)
    
    def smooth_seamline(self, control_points: ControlPoints) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Generate smooth seamline and insert vertices into mesh."""
        # Sample points along the curve
        t_values = np.linspace(0, 1, self.num_samples)
        curve_points = [self._compute_bezier_point(control_points, t) for t in t_values]
        #curve_points, projected_points = self._compute_bezier_point(control_points)
        original_points = [p[0] for p in curve_points]
        projected_points = [p[1] for p in curve_points]
        
        # Save points as PLY files
        self.save_points_as_ply(original_points, "original_curve.ply")
        #self.save_points_as_ply(curve_points, "original_curve.ply")
        self.save_points_as_ply(projected_points, "projected_curve.ply")
        
        # Insert vertices and store their indices
        seam_vertices = []
        for point in projected_points:
            vertex_idx = self._insert_vertex(point)
            seam_vertices.append(vertex_idx)
            
        #return original_points, projected_points, seam_vertices
        return curve_points, projected_points, seam_vertices
    
    def save_mesh(self, output_path: str):
        """Save the modified mesh to a PLY file."""
        self.mesh.export(output_path)
        
    def visualize(self, seam_vertices: List[int] = None):
        """Visualize the mesh with highlighted seamline."""
        scene = trimesh.Scene()
        mesh_visual = self.mesh.copy()
        
        if seam_vertices:
            colors = np.zeros((len(self.mesh.vertices), 4))
            colors[:, 3] = 1.0  # Set alpha to 1
            colors[seam_vertices] = [1.0, 0.0, 0.0, 1.0]  # Red for seam vertices
            mesh_visual.visual.vertex_colors = colors
            
        scene.add_geometry(mesh_visual)
        scene.show()

def main():
    # Example usage
    mesh_path = "data/body/ref.ply"
    output_path = "smooth_seamlines.ply"
    
    mesh: trimesh.Trimesh = trimesh.load_mesh(mesh_path)
    mesh = mesh.subdivide().subdivide()
    
    # Create control points (example coordinates)
    control_points = ControlPoints(
        start=np.array([-0.151695, -0.139629, 0.0636825]),
        control=np.array([-0.214182, -0.615215, 0.010545]),
        end=np.array([-0.266221, -0.998304, -0.0263793])
    )
    
    # Initialize and run seamline smoothing
    smoother = MeshSeamSmoother(mesh)
    seam_vertices = smoother.smooth_seamline(control_points)
    
    # Save result and visualize
    smoother.save_mesh(output_path)
    #smoother.visualize(seam_vertices)

if __name__ == "__main__":
    main()