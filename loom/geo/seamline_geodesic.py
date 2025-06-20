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
    
    def _compute_bezier_point(self, control_points: List) -> Tuple[np.ndarray, np.ndarray]:
        """Compute point on Bézier curve using adapted De Casteljau algorithm."""
        # First level interpolation
        paths = []
        for point_idx in range(len(control_points) - 1):
            paths.append(self._compute_geodesic_path(control_points[point_idx], control_points[point_idx+1]))
    
        return np.vstack(paths)
    
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
    
    def _reposition_closest_vertex(self, point: np.ndarray) -> int:
        """Find and reposition the closest vertex to the target point.
        
        Args:
            point: The target point coordinates
        
        Returns:
            Index of the repositioned vertex
        """
        # Find the closest vertex
        distances = np.linalg.norm(self.mesh.vertices - point, axis=1)
        closest_vertex_idx = np.argmin(distances)
        
        # Update the vertex position
        self.mesh.vertices[closest_vertex_idx] = point
        
        return closest_vertex_idx
    
    def save_points_as_ply(self, points: List[np.ndarray], output_path: str):
        """Save points as PLY point cloud."""
        points_array = np.array(points)
        point_cloud = trimesh.PointCloud(points_array)
        point_cloud.export(output_path)
    
    def smooth_seamline(self, control_points: List) -> None:
        """Generate smooth seamline and insert vertices into mesh."""
        # Sample points along the curve
        curve_points = self._compute_bezier_point(control_points)
        
        curve_points = np.append(curve_points[::2], curve_points[-1]) if len(curve_points) % 2 == 0 else curve_points[::2]
        
        # Save points as PLY files
        self.save_points_as_ply(curve_points, "curve_points.ply")
        
        # Insert vertices and store their indices
        seam_vertices = []
        for point in curve_points:
            vertex_idx = self._reposition_closest_vertex(point)
            #seam_vertices.append(vertex_idx)
    
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
    mesh = mesh#.subdivide()#.subdivide()
    
    # Create control points (example coordinates)
    control_points = np.array([
        #[-0.151695, -0.139629, 0.0636825],     #1
        #[-0.15932, -0.136112, 0.0399444],      #1
        [-0.160295, -0.13227, 0.0142791],       #1
        #[-0.197642, -0.344033, -0.00526804],   #2
        #[-0.214182, -0.615215, 0.010545],      #2
        [-0.266221, -0.998304, -0.0263793]      #3
    ])
    
    # Initialize and run seamline smoothing
    smoother = MeshSeamSmoother(mesh)
    smoother.smooth_seamline(control_points)
    
    # Save result and visualize
    smoother.save_mesh(output_path)
    #smoother.visualize(seam_vertices)

if __name__ == "__main__":
    main()