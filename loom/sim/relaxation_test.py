import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict
import trimesh
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

from loom.const import PATCH_LIST


class MeshVisualizer:
    """
    A class to handle mesh visualization with consistent color scaling across multiple plots.
    This ensures that edge colors represent the same deviation values in both initial and final states.
    """
    def __init__(self, target_length):
        self.target_length = target_length
        self.global_min_deviation = float('inf')
        self.global_max_deviation = float('-inf')
        
    def compute_edge_data(self, vertices, edges):
        """
        Computes edge lengths and deviations for a given mesh state.
        """
        edge_start = vertices[edges[:, 0]]
        edge_end = vertices[edges[:, 1]]
        edge_lengths = np.linalg.norm(edge_end - edge_start, axis=1)
        deviations = (edge_lengths - self.target_length) / self.target_length
        
        # Update global min/max
        self.global_min_deviation = min(self.global_min_deviation, np.min(deviations))
        self.global_max_deviation = max(self.global_max_deviation, np.max(deviations))
        
        return edge_lengths, deviations
    
    def visualize_mesh_edges(self, vertices, faces, edges, title="Mesh Edge Analysis", 
                           save_path=None, show_plot=True):
        """
        Creates a visualization of mesh edges with consistent color scaling.
        """
        # Create figure
        fig = plt.figure(figsize=(15, 8))
        
        # Calculate edge data
        edge_lengths, deviations = self.compute_edge_data(vertices, edges)
        
        # Create custom colormap (blue -> white -> red)
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
        
        # Normalize deviations using global min/max for consistent scaling
        norm_deviations = (deviations - self.global_min_deviation) / (
            self.global_max_deviation - self.global_min_deviation)
        
        # Create 3D subplot
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot edges with consistently scaled colors
        for i, (start_idx, end_idx) in enumerate(edges):
            color = cmap(norm_deviations[i])
            ax1.plot([vertices[start_idx, 0], vertices[end_idx, 0]],
                    [vertices[start_idx, 1], vertices[end_idx, 1]],
                    [vertices[start_idx, 2], vertices[end_idx, 2]],
                    color=color, linewidth=1)
        
        # Set equal aspect ratio and labels
        ax1.set_box_aspect([1, 1, 1])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Edge Length Visualization')
        
        # Add colorbar with global scaling
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(np.linspace(self.global_min_deviation, self.global_max_deviation, 256))
        cbar = plt.colorbar(sm, ax=ax1, label='Length Deviation from Target')
        
        # Add target length marker to colorbar
        cbar.ax.axhline(y=0, color='white', linestyle='-', linewidth=2)
        
        # Create statistics subplot
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        
        # Calculate and display statistics
        stats_text = (
            f'Edge Length Statistics:\n\n'
            f'Total Edges: {len(edges)}\n'
            f'Target Length: {self.target_length:.4f}\n\n'
            f'Current Lengths:\n'
            f'  Mean: {np.mean(edge_lengths):.4f}\n'
            f'  Median: {np.median(edge_lengths):.4f}\n'
            f'  Min: {np.min(edge_lengths):.4f}\n'
            f'  Max: {np.max(edge_lengths):.4f}\n'
            f'  Std Dev: {np.std(edge_lengths):.4f}\n\n'
            f'Constraint Violations:\n'
            f'  Edges > Target: {np.sum(edge_lengths > self.target_length)}\n'
            f'  Max Deviation: {np.max(np.abs(deviations)):.4f}\n'
            f'  Mean Deviation: {np.mean(np.abs(deviations)):.4f}\n\n'
            f'Global Scale Range:\n'
            f'  Min Deviation: {self.global_min_deviation:.4f}\n'
            f'  Max Deviation: {self.global_max_deviation:.4f}'
        )
        
        ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', fontfamily='monospace')
        
        # Set overall title
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()


class GarmentSimulator:
    def __init__(self, body_vertices, body_faces, garment_vertices, garment_faces, edge_ratio=0.95):
        self.body_vertices = np.array(body_vertices)
        self.body_faces = np.array(body_faces)
        self.garment_vertices = np.array(garment_vertices)
        self.garment_faces = np.array(garment_faces)
        
        # Initialize velocities for PBD
        self.velocities = np.zeros_like(self.garment_vertices)
        
        # Precompute mesh data structures
        self._precompute_mesh_structures()
        
        # Extract edges and compute target length
        self.garment_edges = self._extract_edges(garment_faces)
        self.target_length = self._compute_target_length(edge_ratio)
        
        # PBD parameters
        self.damping = 0.8
        self.constraint_stiffness = 0.5
        self.dt = 1.0 / 60.0
        
        # Extract and store boundary vertices
        self.is_boundary = self._extract_boundary_vertices()
        
        # Store initial positions of boundary vertices
        self.boundary_positions = self.garment_vertices[self.is_boundary].copy()
        
        # Add new parameters for adaptive constraint enforcement
        self.max_correction_factor = 0.2  # Increased from 0.1
        self.correction_adaptation_rate = 1.1  # For adaptive stepping
        self.min_correction_factor = 0.01
        
    def _extract_boundary_vertices(self):
        """
        Extracts boundary vertices of the garment mesh using edge analysis.
        
        A boundary vertex is one that lies on an edge that is connected to only
        one triangle. This method builds an edge-to-face mapping and identifies
        edges that appear only once, indicating they are on the boundary.
        
        Returns:
            numpy array: Boolean mask indicating which vertices are on the boundary
        """
        # Create a dictionary to store edge to face mappings
        # We'll use a tuple of vertex indices (sorted for uniqueness) as the key
        edge_to_face = defaultdict(list)
        
        # For each face, add its edges to the mapping
        for face_idx, face in enumerate(self.garment_faces):
            # Get edges of the triangle (in sorted order for consistent lookup)
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            
            # Add face index to each edge's list
            for edge in edges:
                edge_to_face[edge].append(face_idx)
        
        # Find boundary edges (those with only one connected face)
        boundary_edges = [edge for edge, faces in edge_to_face.items() 
                        if len(faces) == 1]
        
        # Create a set of boundary vertices
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)
        
        # Convert to boolean mask for efficient indexing
        is_boundary = np.zeros(len(self.garment_vertices), dtype=bool)
        is_boundary[list(boundary_vertices)] = True
        
        return is_boundary
        
    def _precompute_mesh_structures(self):
        """Precompute all necessary mesh structures for efficient queries."""
        # Compute face normals and centroids
        self.face_normals = self._compute_face_normals()
        self.face_centroids = self._compute_face_centroids()
        
        # Build vertex-face adjacency
        self.vertex_to_faces = defaultdict(list)
        for face_idx, face in enumerate(self.body_faces):
            for vertex_idx in face:
                self.vertex_to_faces[vertex_idx].append(face_idx)
        
        # Build face-face adjacency
        self.face_to_faces = defaultdict(set)
        for vertex_idx, faces in self.vertex_to_faces.items():
            for face1 in faces:
                self.face_to_faces[face1].update(faces)
        
        # Build KD-tree for face centroids
        self.centroid_kdtree = KDTree(self.face_centroids)
        
        # Precompute face basis vectors for faster projection
        self.face_bases = np.zeros((len(self.body_faces), 2, 3))
        for i, face in enumerate(self.body_faces):
            v1, v2, v3 = self.body_vertices[face]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = self.face_normals[i]
            
            # Compute orthonormal basis in triangle plane
            basis1 = edge1 / np.linalg.norm(edge1)
            basis2 = np.cross(normal, basis1)
            self.face_bases[i] = np.vstack((basis1, basis2))
    
    def _compute_face_normals(self):
        """Vectorized computation of face normals."""
        v1 = self.body_vertices[self.body_faces[:, 0]]
        v2 = self.body_vertices[self.body_faces[:, 1]]
        v3 = self.body_vertices[self.body_faces[:, 2]]
        
        normals = np.cross(v2 - v1, v3 - v1)
        lengths = np.linalg.norm(normals, axis=1)
        mask = lengths > 1e-10
        normals[mask] = normals[mask] / lengths[mask, np.newaxis]
        return normals
    
    def _compute_face_centroids(self):
        """Vectorized computation of face centroids."""
        return np.mean(self.body_vertices[self.body_faces], axis=1)
    
    def _extract_edges(self, faces):
        """Extract unique edges using numpy operations."""
        # Create all edges (including duplicates)
        edges = np.vstack((
            np.column_stack((faces[:, 0], faces[:, 1])),
            np.column_stack((faces[:, 1], faces[:, 2])),
            np.column_stack((faces[:, 2], faces[:, 0]))
        ))
        
        # Sort edge vertices and remove duplicates
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        return edges
    
    def _compute_target_length(self, ratio):
        """Vectorized computation of target length."""
        v1 = self.garment_vertices[self.garment_edges[:, 0]]
        v2 = self.garment_vertices[self.garment_edges[:, 1]]
        lengths = np.linalg.norm(v2 - v1, axis=1)
        return np.mean(lengths) * ratio
    
    def _project_to_surface(self, vertices):
        """
        Projects vertices onto the body mesh surface using barycentric coordinates.
        
        Args:
            vertices: numpy array of shape (N, 3) containing vertices to project
            
        Returns:
            projected_vertices: numpy array of shape (N, 3) containing projected positions
        """
        # Find closest face centroids using KD-tree
        _, face_indices = self.centroid_kdtree.query(vertices)
        
        # Get triangle vertices for each closest face
        closest_faces = self.body_faces[face_indices]
        v0 = self.body_vertices[closest_faces[:, 0]]
        v1 = self.body_vertices[closest_faces[:, 1]]
        v2 = self.body_vertices[closest_faces[:, 2]]
        
        # Compute face normals for closest faces
        face_normals = self.face_normals[face_indices]
        
        # Project points onto the triangle planes
        # First, compute vectors from v0 to the query points
        v0_to_point = vertices - v0
        
        # Project these vectors onto the face normal to get the distance to the plane
        dist_to_plane = np.sum(v0_to_point * face_normals, axis=1)
        
        # Subtract this projection to get the point projected onto the plane
        projected_points = vertices - dist_to_plane[:, np.newaxis] * face_normals
        
        # Now compute barycentric coordinates to ensure the point lies within the triangle
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Compute areas using cross products
        normal = np.cross(edge1, edge2)
        area_total = np.linalg.norm(normal, axis=1) / 2
        
        # Compute barycentric coordinates
        p_v0 = projected_points - v0
        p_v1 = projected_points - v1
        p_v2 = projected_points - v2
        
        area0 = np.linalg.norm(np.cross(p_v1, p_v2), axis=1) / 2
        area1 = np.linalg.norm(np.cross(p_v2, p_v0), axis=1) / 2
        area2 = np.linalg.norm(np.cross(p_v0, p_v1), axis=1) / 2
        
        # Convert areas to barycentric coordinates
        b0 = area0 / area_total
        b1 = area1 / area_total
        b2 = area2 / area_total
        
        # Handle points outside the triangle by clamping barycentric coordinates
        sum_coords = b0 + b1 + b2
        b0 /= sum_coords
        b1 /= sum_coords
        b2 /= sum_coords
        
        # Compute final projected positions using barycentric coordinates
        projected_vertices = (b0[:, np.newaxis] * v0 +
                            b1[:, np.newaxis] * v1 +
                            b2[:, np.newaxis] * v2)
        
        return projected_vertices
    
    def _apply_edge_constraints_vectorized(self):
        """Enhanced edge constraint application using direction-based corrections."""
        v1 = self.garment_vertices[self.garment_edges[:, 0]]
        v2 = self.garment_vertices[self.garment_edges[:, 1]]
        
        edge_vectors = v2 - v1
        current_lengths = np.linalg.norm(edge_vectors, axis=1)
        mask = current_lengths > self.target_length
        
        if np.any(mask):
            masked_edges = self.garment_edges[mask]
            
            # Calculate direction-based corrections
            # First normalize the edge vectors to get directions
            directions = edge_vectors[mask] / current_lengths[mask, np.newaxis]
            
            # Calculate length differences directly
            length_differences = current_lengths[mask] - self.target_length
            
            # Calculate correction vectors using directions and length differences
            masked_corrections = directions * length_differences[:, np.newaxis]
            
            # Calculate violation severity for adaptive correction
            #max_violation = np.max(length_differences)
            strain = (current_lengths[mask] - self.target_length) / self.target_length
            correction_factor = np.clip(
                strain / np.max(strain) * self.max_correction_factor,
                self.min_correction_factor,
                self.max_correction_factor
            )
            
            # Scale corrections by adaptive factor and constraint stiffness
            masked_corrections *= correction_factor[:, np.newaxis]
            masked_corrections *= self.constraint_stiffness
            
            # Apply corrections with improved stability
            corrections = np.zeros_like(self.garment_vertices)
            np.add.at(corrections, masked_edges[:, 0], masked_corrections / 2)
            np.add.at(corrections, masked_edges[:, 1], -masked_corrections / 2)
            
            # Count corrections per vertex for weighted averaging
            counts = np.zeros(len(self.garment_vertices))
            np.add.at(counts, masked_edges[:, 0], 1)
            np.add.at(counts, masked_edges[:, 1], 1)
            
            mask = counts > 0
            if np.any(mask):
                corrections[mask] /= counts[mask, np.newaxis]
                
                # Allow larger maximum corrections based on constraint violations
                max_allowed_correction = self.target_length * np.maximum(
                    0.05,  # Base correction limit
                    np.minimum(0.2, np.max(length_differences) / self.target_length)  # Adaptive limit
                )
                
                correction_lengths = np.linalg.norm(corrections, axis=1)
                scale = np.minimum(1.0, max_allowed_correction / np.maximum(correction_lengths, 1e-10))
                corrections *= scale[:, np.newaxis]
                
                # Don't modify boundary vertices
                corrections[self.is_boundary] = 0
                self.garment_vertices += corrections
    
    def simulate(self, num_iterations=100, log_frequency=10):
        """Run the simulation with sub-iterations for stability."""
        # Store original boundary positions for logging
        initial_boundary_positions = self.garment_vertices[self.is_boundary].copy()
        
        # Log initial statistics
        initial_stats = self._compute_edge_statistics()
        print("\nInitial Statistics:")
        print(f"Total edges: {initial_stats['total_edges']}")
        print(f"Violating edges: {initial_stats['violating_edges']} ({initial_stats['violation_percentage']:.2f}%)")
        print(f"Mean deviation: {initial_stats['mean_deviation']:.6f}")
        print(f"Max deviation: {initial_stats['max_deviation']:.6f}")
        print(f"Std deviation: {initial_stats['std_deviation']:.6f}\n")
        
        stats_history = []
        
        # Increase sub-iterations for better convergence
        sub_iterations = 10  # Increased from 5
        
        # Add adaptive relaxation
        current_stiffness = self.constraint_stiffness
        relaxation_factor = 0.95
        
        for iteration in range(num_iterations):
            # Adapt constraint stiffness based on progress
            if iteration > 0 and iteration % 10 == 0:
                current_stats = self._compute_edge_statistics()
                if current_stats['violating_edges'] > stats_history[-1]['violating_edges']:
                    # If violations increased, reduce stiffness
                    current_stiffness *= relaxation_factor
                else:
                    # If violations decreased, carefully increase stiffness
                    current_stiffness = min(1.0, current_stiffness * (1 / relaxation_factor))
            
            for sub_iter in range(sub_iterations):
                prev_positions = self.garment_vertices.copy()
                
                # Apply edge constraints with current stiffness
                old_stiffness = self.constraint_stiffness
                self.constraint_stiffness = current_stiffness
                self._apply_edge_constraints_vectorized()
                self.constraint_stiffness = old_stiffness
                
                # Project to surface with weighted blending
                projected_positions = self._project_to_surface(self.garment_vertices)
                non_boundary_mask = ~self.is_boundary
                
                # Adaptive surface projection weight
                surface_weight = 0.3 * (1 - (iteration / num_iterations))  # Decrease over time
                self.garment_vertices[non_boundary_mask] = (
                    (1 - surface_weight) * self.garment_vertices[non_boundary_mask] +
                    surface_weight * projected_positions[non_boundary_mask]
                )
                
                # Maintain boundary vertices
                self.garment_vertices[self.is_boundary] = self.boundary_positions
                
                # Update velocities with adaptive damping
                self.velocities[non_boundary_mask] = (
                    (self.garment_vertices[non_boundary_mask] - 
                     prev_positions[non_boundary_mask]) / self.dt
                )
                adaptive_damping = self.damping ** (1.0 / (sub_iter + 1))  # Stronger damping in early sub-iterations
                self.velocities[non_boundary_mask] *= adaptive_damping
                
                # Zero boundary velocities
                self.velocities[self.is_boundary] = 0
            
            # Log progress at specified frequency
            if iteration % log_frequency == 0 or iteration == num_iterations - 1:
                current_stats = self._compute_edge_statistics()
                stats_history.append(current_stats)
                
                print(f"\nIteration {iteration + 1}/{num_iterations}:")
                print(f"Violating edges: {current_stats['violating_edges']} ({current_stats['violation_percentage']:.2f}%)")
                print(f"Mean deviation: {current_stats['mean_deviation']:.6f}")
                print(f"Max deviation: {current_stats['max_deviation']:.6f}")
                print(f"Std deviation: {current_stats['std_deviation']:.6f}")
        
        # Print final improvement summary
        final_stats = stats_history[-1]
        improvement = {
            'violation_reduction': initial_stats['violating_edges'] - final_stats['violating_edges'],
            'violation_percentage_reduction': initial_stats['violation_percentage'] - final_stats['violation_percentage'],
            'mean_deviation_reduction': initial_stats['mean_deviation'] - final_stats['mean_deviation'],
            'max_deviation_reduction': initial_stats['max_deviation'] - final_stats['max_deviation']
        }
        
        print("\nFinal Improvement Summary:")
        print(f"Violation count reduction: {improvement['violation_reduction']} edges")
        print(f"Violation percentage reduction: {improvement['violation_percentage_reduction']:.2f}%")
        print(f"Mean deviation reduction: {improvement['mean_deviation_reduction']:.6f}")
        print(f"Max deviation reduction: {improvement['max_deviation_reduction']:.6f}\n")
        
        return self.garment_vertices, stats_history
    
    def _compute_edge_statistics(self):
        """Compute statistics about edge length constraints."""
        v1 = self.garment_vertices[self.garment_edges[:, 0]]
        v2 = self.garment_vertices[self.garment_edges[:, 1]]
        
        current_lengths = np.linalg.norm(v2 - v1, axis=1)
        violations = current_lengths > self.target_length
        
        stats = {
            'total_edges': len(self.garment_edges),
            'violating_edges': np.sum(violations),
            'violation_percentage': (np.sum(violations) / len(self.garment_edges)) * 100
        }
        
        if np.any(violations):
            deviations = current_lengths[violations] - self.target_length
            stats.update({
                'max_deviation': np.max(deviations),
                'min_deviation': np.min(deviations),
                'mean_deviation': np.mean(deviations),
                'median_deviation': np.median(deviations),
                'std_deviation': np.std(deviations)
            })
        else:
            stats.update({
                'max_deviation': 0,
                'min_deviation': 0,
                'mean_deviation': 0,
                'median_deviation': 0,
                'std_deviation': 0
            })
        
        return stats


def run_simulation(body_vertices, body_faces, garment_vertices, garment_faces, 
                   edge_ratio=0.95, num_iterations=100, 
                  log_frequency=10):
    """
    Run simulation with consistently scaled visualizations.
    """
    simulator = GarmentSimulator(body_vertices, body_faces, 
                               garment_vertices, garment_faces, 
                               edge_ratio)
    
    # Create visualizer with target length
    visualizer = MeshVisualizer(simulator.target_length)
    
    # Compute initial edge data to establish global scale
    visualizer.compute_edge_data(simulator.garment_vertices, simulator.garment_edges)
    
    # Visualize initial state
    print("\nInitial State Analysis:")
    visualizer.visualize_mesh_edges(
        simulator.garment_vertices,
        simulator.garment_faces,
        simulator.garment_edges,
        title="Initial State",
        save_path="initial_state.png"
    )
    
    # Run simulation
    final_vertices, stats_history = simulator.simulate(num_iterations, log_frequency)
    
    # Visualize final state using same scale
    print("\nFinal State Analysis:")
    visualizer.visualize_mesh_edges(
        simulator.garment_vertices,
        simulator.garment_faces,
        simulator.garment_edges,
        title="Final State",
        save_path="final_state.png"
    )
    
    return final_vertices, stats_history


if __name__ == '__main__':
    body_mesh = trimesh.load('data/body/ref.ply')
    
    mesh_3d_dir = os.path.join('/Users/kristijanbartol/TailorLang/data/embedded/')
    upper_patch_meshes = []
    lower_patch_meshes = []
    
    for patch_label in PATCH_LIST:
        patch_mesh = trimesh.load(os.path.join(mesh_3d_dir, patch_label, 'ref.ply'))
        if 'lower' in patch_label:
            lower_patch_meshes.append(patch_mesh)
        else:
            upper_patch_meshes.append(patch_mesh)
        
    upper_mesh = trimesh.util.concatenate(upper_patch_meshes)
    lower_mesh = trimesh.util.concatenate(lower_patch_meshes)
    
    upper_mesh = trimesh.Trimesh(vertices=upper_mesh.vertices, faces=upper_mesh.faces, process=True)
    upper_mesh.export('before.ply')

    final_verts = run_simulation(
        body_vertices=body_mesh.vertices,
        body_faces=body_mesh.faces,
        garment_vertices=upper_mesh.vertices,
        garment_faces=upper_mesh.faces,
        edge_ratio=1.2,
        num_iterations=100,
        log_frequency=10
    )
    
    trimesh.Trimesh(vertices=final_verts[0], faces=upper_mesh.faces).export('relaxed.ply')
    