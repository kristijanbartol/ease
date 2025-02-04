import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

class GridClothSimulator:
    def __init__(self, rows=10, cols=10, spacing=1.0):
        """
        Initialize a 2D grid cloth simulator.
        
        Args:
            rows (int): Number of rows in the grid
            cols (int): Number of columns in the grid
            spacing (float): Initial spacing between vertices
        """
        self.rows = rows
        self.cols = cols
        self.spacing = spacing
        
        # Initialize grid vertices
        x = np.linspace(0, (cols-1)*spacing, cols)
        y = np.linspace(0, (rows-1)*spacing, rows)
        X, Y = np.meshgrid(x, y)
        self.vertices = np.stack([X.flatten(), Y.flatten()], axis=1)
        
        # Initialize velocities
        self.velocities = np.zeros_like(self.vertices)
        
        # Create edges (horizontal and vertical)
        self.edges = self._create_edges()
        
        # Initialize edge constraints
        self.edge_rest_lengths = np.linalg.norm(
            self.vertices[self.edges[:, 1]] - self.vertices[self.edges[:, 0]], 
            axis=1
        )
        
        # Mark boundary vertices (vertices on the grid edges)
        self.is_boundary = self._identify_boundary_vertices()
        
        # Initialize simulation parameters
        self.damping = 0.8
        self.constraint_stiffness = 0.3
        self.dt = 1.0 / 60.0
        
        # Initialize edge constraints dictionary
        self.constrained_edges = {}  # Will store {edge_idx: target_length_ratio}
        
    def _create_edges(self):
        """Create horizontal and vertical edges for the grid."""
        edges = []
        
        # Horizontal edges
        for i in range(self.rows):
            for j in range(self.cols - 1):
                idx1 = i * self.cols + j
                idx2 = i * self.cols + (j + 1)
                edges.append([idx1, idx2])
                
        # Vertical edges
        for i in range(self.rows - 1):
            for j in range(self.cols):
                idx1 = i * self.cols + j
                idx2 = (i + 1) * self.cols + j
                edges.append([idx1, idx2])
                
        return np.array(edges)
    
    def _identify_boundary_vertices(self):
        """Identify vertices on the grid boundaries."""
        is_boundary = np.zeros(len(self.vertices), dtype=bool)
        
        # Mark top and bottom rows
        is_boundary[:self.cols] = True  # Top row
        is_boundary[-self.cols:] = True  # Bottom row
        
        # Mark left and right columns
        left_col = np.arange(0, self.rows * self.cols, self.cols)
        right_col = np.arange(self.cols - 1, self.rows * self.cols, self.cols)
        is_boundary[left_col] = True
        is_boundary[right_col] = True
        
        return is_boundary
    
    def add_edge_constraint(self, edge_idx, target_length_ratio):
        """
        Add a constraint to a specific edge.
        
        Args:
            edge_idx (int): Index of the edge to constrain
            target_length_ratio (float): Target length as a ratio of original length
        """
        self.constrained_edges[edge_idx] = target_length_ratio
    
    def _apply_edge_constraints(self):
        """Apply edge length constraints."""
        # Get current edge vectors and lengths
        edge_vectors = self.vertices[self.edges[:, 1]] - self.vertices[self.edges[:, 0]]
        current_lengths = np.linalg.norm(edge_vectors, axis=1)
        
        # Initialize corrections
        corrections = np.zeros_like(self.vertices)
        correction_counts = np.zeros(len(self.vertices))
        
        # Process all edges
        for i, (v1_idx, v2_idx) in enumerate(self.edges):
            # Get target length for this edge
            target_length = self.edge_rest_lengths[i]
            if i in self.constrained_edges:
                target_length *= self.constrained_edges[i]
            
            # Calculate correction
            current_length = current_lengths[i]
            if abs(current_length - target_length) > 1e-6:
                correction = edge_vectors[i] * (1 - target_length/current_length)
                correction *= self.constraint_stiffness
                
                # Apply correction to both vertices
                corrections[v1_idx] += correction / 2
                corrections[v2_idx] -= correction / 2
                
                # Count corrections per vertex
                correction_counts[v1_idx] += 1
                correction_counts[v2_idx] += 1
        
        # Average corrections and apply them
        mask = correction_counts > 0
        corrections[mask] /= correction_counts[mask, np.newaxis]
        
        # Don't move boundary vertices
        corrections[self.is_boundary] = 0
        
        # Apply corrections
        self.vertices += corrections
    
    def simulate(self, num_iterations=100, log_every=1, ineq_ratio_threshold=1.1):
        """Run the simulation for specified number of iterations."""
        for iteration in range(num_iterations):
            # Store previous positions
            prev_positions = self.vertices.copy()
            
            # Apply constraints
            self._apply_edge_constraints()
            
            # Update velocities (not applied to boundary vertices)
            non_boundary = ~self.is_boundary
            self.velocities[non_boundary] = (
                (self.vertices[non_boundary] - prev_positions[non_boundary]) / self.dt
            )
            
            # Apply damping
            self.velocities *= self.damping
            
            # Log statistics every N iterations
            if iteration % log_every == 0:
                # Calculate current edge lengths
                edge_vectors = self.vertices[self.edges[:, 1]] - self.vertices[self.edges[:, 0]]
                current_lengths = np.linalg.norm(edge_vectors, axis=1)
                
                # Calculate length ratios compared to rest lengths
                length_ratios = current_lengths / self.edge_rest_lengths
                
                # Find maximum edge length ratio
                max_ratio = length_ratios.max()
                max_ratio_idx = length_ratios.argmax()
                
                # Find edges with significant deviation (>= 1% from original length)
                EPS = 1e-4
                significant_deviations = length_ratios > ineq_ratio_threshold + EPS
                if np.any(significant_deviations):
                    significant_ratios = length_ratios[significant_deviations]
                    mean_deviation = np.mean(significant_ratios)
                    median_deviation = np.median(significant_ratios)
                    num_deviating = np.sum(significant_deviations)
                    
                    print(f"\nIteration {iteration}:")
                    print(f"Max length ratio: {max_ratio:.3f} (edge {max_ratio_idx})")
                    print(f"Number of edges deviating ≥{(ineq_ratio_threshold - 1) * 100}%: {num_deviating}")
                    print(f"Mean ratio of deviating edges: {mean_deviation:.3f}")
                    print(f"Median ratio of deviating edges: {median_deviation:.3f}")
                else:
                    print(f"\nIteration {iteration}: No edges deviating ≥{(ineq_ratio_threshold - 1) * 100:.1f}% from original length")
    
    def visualize(self, title="Grid Cloth Simulation"):
        """Visualize the current state of the grid."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw edges with color based on stretch/compression
        edge_vectors = self.vertices[self.edges[:, 1]] - self.vertices[self.edges[:, 0]]
        current_lengths = np.linalg.norm(edge_vectors, axis=1)
        strain = (current_lengths - self.edge_rest_lengths) / self.edge_rest_lengths
        
        # Create line segments for visualization
        segments = np.stack([self.vertices[self.edges[:, 0]], 
                           self.vertices[self.edges[:, 1]]], axis=1)
        
        # Create custom colormap (blue -> white -> red)
        colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
        
        # Create line collection with colors based on strain
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(-0.2, 0.2))
        lc.set_array(strain)
        
        # Add lines to plot
        ax.add_collection(lc)
        
        # Plot vertices
        ax.plot(self.vertices[:, 0], self.vertices[:, 1], 'k.', markersize=5)
        
        # Highlight boundary vertices
        ax.plot(self.vertices[self.is_boundary, 0], 
                self.vertices[self.is_boundary, 1], 
                'r.', markersize=8)
        
        # Highlight constrained edges
        if self.constrained_edges:
            for edge_idx in self.constrained_edges:
                v1, v2 = self.edges[edge_idx]
                ax.plot([self.vertices[v1, 0], self.vertices[v2, 0]],
                       [self.vertices[v1, 1], self.vertices[v2, 1]],
                       'g-', linewidth=2, alpha=0.5)
        
        # Add colorbar
        plt.colorbar(lc, ax=ax, label='Strain (+ = stretch, - = compress)')
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Set limits with some padding
        padding = self.spacing
        ax.set_xlim(self.vertices[:, 0].min() - padding, 
                   self.vertices[:, 0].max() + padding)
        ax.set_ylim(self.vertices[:, 1].min() - padding, 
                   self.vertices[:, 1].max() + padding)
        
        plt.savefig('init.png', dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    GRID_SIZE = 25
    NROWS = GRID_SIZE
    NCOLS = GRID_SIZE
    NEDGES = (NROWS - 1) * NCOLS * 2
    
    INEQ_RATIO = 1.09
    
    # Create simulator with a 10x10 grid
    simulator = GridClothSimulator(rows=NROWS, cols=NCOLS, spacing=1.0)

    eq_edge_idxs = [15, 24, 33, 121, 135, 147]
    
    # Add a constraint to make one edge shorter
    for edge_idx in eq_edge_idxs:
        simulator.add_edge_constraint(edge_idx, 0.6)
    
    # Visualize initial state
    simulator.visualize("Initial State")
    
    # Run simulation
    simulator.simulate(num_iterations=50)
    
    # Visualize final state
    simulator.visualize("Final State")