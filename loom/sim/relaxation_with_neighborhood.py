import numpy as np
import trimesh
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from collections import defaultdict

class GarmentRelaxer:
    def __init__(self, body_mesh, garment_tpose, garment_apose, param_mesh, sub_iterations=10):
        """
        Initialize the garment relaxation system.
        
        Args:
            body_mesh: Trimesh object of the body in A-pose
            garment_tpose: Trimesh object of the garment in T-pose
            garment_apose: Trimesh object of the garment in A-pose
            param_mesh: Trimesh object of the 2D parameterization mesh
            sub_iterations: Number of sub-iterations for constraint satisfaction
        """
        self.body_mesh = body_mesh
        self.garment_tpose = garment_tpose
        self.garment_apose = garment_apose
        self.param_mesh = param_mesh
        self.sub_iterations = sub_iterations
        
        # Verify mesh compatibility
        assert len(garment_tpose.vertices) == len(garment_apose.vertices), "Garment meshes must have same number of vertices"
        assert len(garment_tpose.faces) == len(garment_apose.faces), "Garment meshes must have same number of faces"
        assert len(param_mesh.vertices) == len(garment_tpose.vertices), "Param mesh must match garment topology"
        
        # Precompute mesh structures for efficient surface projection
        self._precompute_mesh_structures()
        
        # Extract edges and compute target lengths from T-pose
        self.edges = self._extract_edges(garment_tpose.faces)
        self.target_lengths = self._compute_edge_lengths(garment_tpose.vertices, self.edges)
        
        # Initialize current state
        self.current_vertices = garment_apose.vertices.copy()
        self.velocities = np.zeros_like(self.current_vertices)
        
        # Extract boundary vertices
        self.is_boundary = self._extract_boundary_vertices()
        self.boundary_positions = garment_apose.vertices[self.is_boundary].copy()
        
        # Initialize simulation parameters
        self.damping = 0.8
        self.constraint_stiffness = 0.3
        self.dt = 1.0 / 60.0
        self.max_correction_factor = 0.2
        self.min_correction_factor = 0.01
        
    def _precompute_mesh_structures(self):
        """Precompute necessary mesh structures for efficient surface projection."""
        # Compute face normals and centroids for body mesh
        self.face_normals = self._compute_face_normals()
        self.face_centroids = self._compute_face_centroids()
        
        # Build vertex-face adjacency
        self.vertex_to_faces = defaultdict(list)
        for face_idx, face in enumerate(self.body_mesh.faces):
            for vertex_idx in face:
                self.vertex_to_faces[vertex_idx].append(face_idx)
        
        # Build face-face adjacency
        self.face_to_faces = defaultdict(set)
        for vertex_idx, faces in self.vertex_to_faces.items():
            for face1 in faces:
                self.face_to_faces[face1].update(faces)
        
        # Build KD-tree for face centroids
        self.centroid_kdtree = KDTree(self.face_centroids)
        
        # Precompute face basis vectors
        self.face_bases = np.zeros((len(self.body_mesh.faces), 2, 3))
        for i, face in enumerate(self.body_mesh.faces):
            v1, v2, v3 = self.body_mesh.vertices[face]
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = self.face_normals[i]
            
            # Compute orthonormal basis in triangle plane
            basis1 = edge1 / np.linalg.norm(edge1)
            basis2 = np.cross(normal, basis1)
            self.face_bases[i] = np.vstack((basis1, basis2))
    
    def _compute_face_normals(self):
        """Compute face normals for body mesh."""
        v1 = self.body_mesh.vertices[self.body_mesh.faces[:, 0]]
        v2 = self.body_mesh.vertices[self.body_mesh.faces[:, 1]]
        v3 = self.body_mesh.vertices[self.body_mesh.faces[:, 2]]
        
        normals = np.cross(v2 - v1, v3 - v1)
        lengths = np.linalg.norm(normals, axis=1)
        mask = lengths > 1e-10
        normals[mask] = normals[mask] / lengths[mask, np.newaxis]
        return normals
    
    def _compute_face_centroids(self):
        """Compute face centroids for body mesh."""
        return np.mean(self.body_mesh.vertices[self.body_mesh.faces], axis=1)
    
    def _project_to_surface(self, vertices):
        """Project vertices onto the body mesh surface using barycentric coordinates."""
        # Find closest face centroids using KD-tree
        _, face_indices = self.centroid_kdtree.query(vertices)
        
        # Get triangle vertices for each closest face
        closest_faces = self.body_mesh.faces[face_indices]
        v0 = self.body_mesh.vertices[closest_faces[:, 0]]
        v1 = self.body_mesh.vertices[closest_faces[:, 1]]
        v2 = self.body_mesh.vertices[closest_faces[:, 2]]
        
        # Get face normals for closest faces
        face_normals = self.face_normals[face_indices]
        
        # Project points onto triangle planes
        v0_to_point = vertices - v0
        dist_to_plane = np.sum(v0_to_point * face_normals, axis=1)
        projected_points = vertices - dist_to_plane[:, np.newaxis] * face_normals
        
        # Compute barycentric coordinates
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Compute areas using cross products
        normal = np.cross(edge1, edge2)
        area_total = np.linalg.norm(normal, axis=1) / 2
        
        p_v0 = projected_points - v0
        p_v1 = projected_points - v1
        p_v2 = projected_points - v2
        
        area0 = np.linalg.norm(np.cross(p_v1, p_v2), axis=1) / 2
        area1 = np.linalg.norm(np.cross(p_v2, p_v0), axis=1) / 2
        area2 = np.linalg.norm(np.cross(p_v0, p_v1), axis=1) / 2
        
        # Convert to barycentric coordinates
        b0 = area0 / area_total
        b1 = area1 / area_total
        b2 = area2 / area_total
        
        # Handle points outside triangle
        sum_coords = b0 + b1 + b2
        b0 /= sum_coords
        b1 /= sum_coords
        b2 /= sum_coords
        
        # Compute final projected positions
        projected_vertices = (b0[:, np.newaxis] * v0 +
                            b1[:, np.newaxis] * v1 +
                            b2[:, np.newaxis] * v2)
        
        return projected_vertices
    
    def _extract_edges(self, faces):
        """Extract unique edges from triangle faces."""
        edges = np.vstack((
            np.column_stack((faces[:, 0], faces[:, 1])),
            np.column_stack((faces[:, 1], faces[:, 2])),
            np.column_stack((faces[:, 2], faces[:, 0]))
        ))
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        return edges
    
    def _compute_edge_lengths(self, vertices, edges):
        """Compute edge lengths."""
        return np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    
    def _extract_boundary_vertices(self):
        """Extract boundary vertices using edge analysis."""
        edge_to_face = {}
        
        for face_idx, face in enumerate(self.garment_apose.faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                if edge in edge_to_face:
                    edge_to_face[edge].append(face_idx)
                else:
                    edge_to_face[edge] = [face_idx]
        
        boundary_vertices = set()
        for edge, faces in edge_to_face.items():
            if len(faces) == 1:
                boundary_vertices.update(edge)
        
        is_boundary = np.zeros(len(self.current_vertices), dtype=bool)
        is_boundary[list(boundary_vertices)] = True
        return is_boundary
    
    def _apply_edge_constraints(self):
        """Apply edge length constraints with bidirectional strain handling."""
        for _ in range(self.sub_iterations):
            # Get current edge vectors and lengths
            edge_vectors = self.current_vertices[self.edges[:, 1]] - self.current_vertices[self.edges[:, 0]]
            current_lengths = np.linalg.norm(edge_vectors, axis=1)
            
            # Calculate strain for all edges (both compression and stretch)
            strain = (current_lengths - self.target_lengths) / self.target_lengths
            
            # Sort edges by absolute strain magnitude
            edge_order = np.argsort(np.abs(strain))[::-1]
            
            # Initialize corrections
            corrections = np.zeros_like(self.current_vertices)
            correction_weights = np.zeros(len(self.current_vertices))
            
            # Process edges in order of strain magnitude
            for edge_idx in edge_order:
                v1_idx, v2_idx = self.edges[edge_idx]
                
                # Skip if both vertices are boundary
                if self.is_boundary[v1_idx] and self.is_boundary[v2_idx]:
                    continue
                
                current_length = current_lengths[edge_idx]
                target_length = self.target_lengths[edge_idx]
                edge_strain = strain[edge_idx]
                
                # Calculate correction direction
                direction = edge_vectors[edge_idx] / current_length
                
                # Different handling for compression vs stretch
                if current_length > target_length:
                    # Stretching case
                    correction = direction * (current_length - target_length)
                    weight = self.constraint_stiffness * (1.0 + 5.0 * abs(edge_strain))
                else:
                    # Compression case - softer correction
                    compression_ratio = abs(edge_strain)
                    if compression_ratio > 0.1:  # Only correct significant compressions
                        correction = direction * (current_length - target_length)
                        weight = self.constraint_stiffness * compression_ratio * 0.5
                    else:
                        continue
                
                # Apply weighted correction
                if not self.is_boundary[v1_idx]:
                    corrections[v1_idx] += correction * weight / 2
                    correction_weights[v1_idx] += weight
                
                if not self.is_boundary[v2_idx]:
                    corrections[v2_idx] -= correction * weight / 2
                    correction_weights[v2_idx] += weight
            
            # Apply weighted corrections
            mask = correction_weights > 0
            corrections[mask] /= correction_weights[mask, np.newaxis]
            
            # Scale corrections based on local neighborhood average strain
            neighborhood_strain = self._compute_neighborhood_strain()
            max_movement = np.mean(self.target_lengths) * 0.1
            correction_magnitudes = np.linalg.norm(corrections, axis=1)
            
            # Adjust scale factors based on neighborhood strain
            scale_factors = np.minimum(1.0, max_movement / (correction_magnitudes + 1e-10))
            scale_factors *= (1.0 + neighborhood_strain)  # Allow larger movements in high-strain regions
            
            corrections *= scale_factors[:, np.newaxis]
            
            # Apply corrections
            self.current_vertices += corrections
            
            # Project non-boundary vertices to surface
            non_boundary = ~self.is_boundary
            self.current_vertices[non_boundary] = self._project_to_surface(
                self.current_vertices[non_boundary]
            )

    def _compute_neighborhood_strain(self):
        """Compute average strain in vertex neighborhoods."""
        vertex_strain = np.zeros(len(self.current_vertices))
        vertex_counts = np.zeros(len(self.current_vertices))
        
        # Get current edge strains
        edge_vectors = self.current_vertices[self.edges[:, 1]] - self.current_vertices[self.edges[:, 0]]
        current_lengths = np.linalg.norm(edge_vectors, axis=1)
        strain = abs((current_lengths - self.target_lengths) / self.target_lengths)
        
        # Accumulate strain for each vertex
        for i, (v1, v2) in enumerate(self.edges):
            vertex_strain[v1] += strain[i]
            vertex_strain[v2] += strain[i]
            vertex_counts[v1] += 1
            vertex_counts[v2] += 1
        
        # Average the strain
        mask = vertex_counts > 0
        vertex_strain[mask] /= vertex_counts[mask]
        
        return vertex_strain
    
    def simulate(self, num_iterations=100, log_frequency=10):
        """Run the simulation with surface projection."""
        # Store initial statistics
        initial_stats = self._compute_statistics()
        print("\nInitial Statistics:")
        self._print_statistics(initial_stats)
        
        stats_history = []
        
        # Adaptive relaxation parameters
        current_stiffness = self.constraint_stiffness
        relaxation_factor = 0.95
        
        for iteration in range(num_iterations):
            # Adapt constraint stiffness
            if iteration > 0 and iteration % 10 == 0:
                current_stats = self._compute_statistics()
                if len(stats_history) > 0 and current_stats['num_violations'] > stats_history[-1]['num_violations']:
                    current_stiffness *= relaxation_factor
                else:
                    current_stiffness = min(1.0, current_stiffness * (1 / relaxation_factor))
            
            # Store previous positions
            prev_positions = self.current_vertices.copy()
            
            # Apply edge constraints with current stiffness
            old_stiffness = self.constraint_stiffness
            self.constraint_stiffness = current_stiffness
            self._apply_edge_constraints()
            self.constraint_stiffness = old_stiffness
            
            # Update velocities and positions
            non_boundary = ~self.is_boundary
            self.velocities[non_boundary] = (
                (self.current_vertices[non_boundary] - prev_positions[non_boundary]) / self.dt
            )
            
            # Adaptive damping
            progress = iteration / num_iterations
            adaptive_damping = self.damping * (1.0 - 0.5 * progress)
            self.velocities *= adaptive_damping
            
            # Apply velocity updates
            self.current_vertices[non_boundary] += self.velocities[non_boundary] * self.dt
            
            # Project to surface
            self.current_vertices[non_boundary] = self._project_to_surface(
                self.current_vertices[non_boundary]
            )
            
            # Maintain boundary vertices
            self.current_vertices[self.is_boundary] = self.boundary_positions
            
            # Log progress
            if iteration % log_frequency == 0:
                current_stats = self._compute_statistics()
                stats_history.append(current_stats)
                print(f"\nIteration {iteration}:")
                self._print_statistics(current_stats)
        
        # Print improvement summary
        final_stats = stats_history[-1]
        self._print_improvement_summary(initial_stats, final_stats)

    def _compute_statistics(self):
        """Compute current edge length statistics."""
        current_lengths = self._compute_edge_lengths(self.current_vertices, self.edges)
        strain = (current_lengths - self.target_lengths) / self.target_lengths
        violations = current_lengths > self.target_lengths
        
        stats = {
            'total_edges': len(self.edges),
            'num_violations': np.sum(violations),
            'violation_percentage': (np.sum(violations) / len(self.edges)) * 100
        }
        
        if np.any(violations):
            violation_strains = strain[violations]
            stats.update({
                'max_strain': np.max(violation_strains),
                'min_strain': np.min(violation_strains),
                'mean_strain': np.mean(violation_strains),
                'median_strain': np.median(violation_strains),
                'std_strain': np.std(violation_strains)
            })
        else:
            stats.update({
                'max_strain': 0,
                'min_strain': 0,
                'mean_strain': 0,
                'median_strain': 0,
                'std_strain': 0
            })
        
        return stats
    
    def _print_statistics(self, stats):
        """Print current statistics in a formatted way."""
        print(f"Total edges: {stats['total_edges']}")
        print(f"Violations: {stats['num_violations']} ({stats['violation_percentage']:.2f}%)")
        print(f"Strain statistics for violating edges:")
        print(f"  Max: {stats['max_strain']:.6f}")
        print(f"  Mean: {stats['mean_strain']:.6f}")
        print(f"  Median: {stats['median_strain']:.6f}")
        print(f"  Std Dev: {stats['std_strain']:.6f}")
    
    def _print_improvement_summary(self, initial_stats, final_stats):
        """Print summary of improvements from initial to final state."""
        print("\nImprovement Summary:")
        print(f"Violation reduction: {initial_stats['num_violations'] - final_stats['num_violations']} edges")
        print(f"Violation percentage reduction: {initial_stats['violation_percentage'] - final_stats['violation_percentage']:.2f}%")
        
        if initial_stats['num_violations'] > 0:
            print("\nStrain reduction:")
            print(f"  Max strain: {initial_stats['max_strain'] - final_stats['max_strain']:.6f}")
            print(f"  Mean strain: {initial_stats['mean_strain'] - final_stats['mean_strain']:.6f}")
    
    def visualize_param_mesh(self, title="Edge Length Analysis", save_path=None, min_strain=None, max_strain=None):
        """Visualize edge strains on the 2D parameterization mesh."""
        # Compute current edge lengths and strain
        current_lengths = self._compute_edge_lengths(self.current_vertices, self.edges)
        #strain = (current_lengths - self.target_lengths) / self.target_lengths
        strain = (current_lengths - self.target_lengths)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Create line segments for visualization using param mesh coordinates
        segments = np.stack([
            self.param_mesh.vertices[self.edges[:, 0], :2],
            self.param_mesh.vertices[self.edges[:, 1], :2]
        ], axis=1)
        
        # Create custom colormap (blue -> white -> red)
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
        
        # Create line collection
        #lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(-0.2, 0.2))
        if min_strain and max_strain:
            lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(min_strain, max_strain))
        else:
            lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(strain.min(), strain.max()))
        lc.set_array(strain)
        
        # Plot edge strains
        ax1.add_collection(lc)
        ax1.plot(self.param_mesh.vertices[:, 0], 
                 self.param_mesh.vertices[:, 1], 
                 'k.', markersize=2)
        ax1.plot(self.param_mesh.vertices[self.is_boundary, 0],
                 self.param_mesh.vertices[self.is_boundary, 1],
                 'r.', markersize=4)
        
        # Add colorbar
        plt.colorbar(lc, ax=ax1, label='Strain (+ = stretch, - = compress)')
        
        # Set plot properties
        ax1.set_aspect('equal')
        ax1.set_title("Edge Strain Visualization")
        
        # Add histogram of strains
        violations = strain > 0
        if np.any(violations):
            if max_strain:
                ax2.hist(strain[violations], bins=50, range=(0, max_strain))
            else:
                ax2.hist(strain[violations], bins=50, range=(0, np.max(strain)))
            ax2.set_title("Strain Distribution (Violations Only)")
            ax2.set_xlabel("Strain")
            ax2.set_ylabel("Count")
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "No violations!", 
                    horizontalalignment='center',
                    verticalalignment='center')
        
        # Set overall title
        fig.suptitle(title, fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return strain.min(), strain.max()


def main():
    """Main execution function."""
    # Load meshes
    body_mesh = trimesh.load('data/body/target-00.ply')
    garment_tpose = trimesh.load('data/embedded/lower_back_right/ref.ply')
    garment_apose = trimesh.load('data/embedded/lower_back_right/target-00.ply')
    param_mesh = trimesh.load('results/pattern/latest/lower_back_right/optim_final-seams.ply')
    
    # Create relaxer
    relaxer = GarmentRelaxer(
        body_mesh=body_mesh,
        garment_tpose=garment_tpose,
        garment_apose=garment_apose,
        param_mesh=param_mesh,
        sub_iterations=10
    )
    
    # Visualize initial state
    min_strain, max_strain = relaxer.visualize_param_mesh("Initial State", "initial_state_with_neighborhood.png")
    
    # Run simulation
    relaxer.simulate(num_iterations=25, log_frequency=10)
    
    # Visualize final state
    relaxer.visualize_param_mesh("Final State", "final_state_with_neighborhood.png", min_strain, max_strain)
    
    # Save final mesh
    final_mesh = trimesh.Trimesh(
        vertices=relaxer.current_vertices,
        faces=garment_apose.faces
    )
    final_mesh.export("relaxed_garment.ply")

if __name__ == "__main__":
    main()