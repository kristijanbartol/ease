import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

import random


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
        """Apply edge length constraints with improved propagation."""
        # Number of sub-iterations for constraint satisfaction
        sub_iterations = 10
        
        for _ in range(sub_iterations):
            # Get current edge vectors and lengths
            edge_vectors = self.vertices[self.edges[:, 1]] - self.vertices[self.edges[:, 0]]
            current_lengths = np.linalg.norm(edge_vectors, axis=1)
            
            # Calculate strain for all edges
            strain = (current_lengths - self.edge_rest_lengths) / self.edge_rest_lengths
            
            # Sort edges by strain magnitude to handle most strained edges first
            edge_order = np.argsort(np.abs(strain))[::-1]
            
            # Initialize corrections
            corrections = np.zeros_like(self.vertices)
            correction_weights = np.zeros(len(self.vertices))
            
            # Process edges in order of strain magnitude
            for edge_idx in edge_order:
                v1_idx, v2_idx = self.edges[edge_idx]
                
                # Skip if both vertices are boundary
                if self.is_boundary[v1_idx] and self.is_boundary[v2_idx]:
                    continue
                
                # Get target length for this edge
                target_length = self.edge_rest_lengths[edge_idx]
                if edge_idx in self.constrained_edges:
                    target_length *= self.constrained_edges[edge_idx]
                    stiffness = 1.0  # Higher stiffness for constrained edges
                else:
                    stiffness = self.constraint_stiffness
                
                # Calculate strain-based correction
                current_length = current_lengths[edge_idx]
                if abs(current_length - target_length) > 1e-6:
                    direction = edge_vectors[edge_idx] / current_length
                    correction = direction * (current_length - target_length)
                    
                    # Weight correction based on distance from constrained edges
                    weight = stiffness * (1.0 + 5.0 * abs(strain[edge_idx]))
                    
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
            
            # Scale corrections based on maximum allowed movement
            max_movement = self.spacing * 0.1  # Limit per sub-iteration
            correction_magnitudes = np.linalg.norm(corrections, axis=1)
            scale_factors = np.minimum(1.0, max_movement / (correction_magnitudes + 1e-10))
            corrections *= scale_factors[:, np.newaxis]
            
            # Apply corrections
            self.vertices += corrections
            
            # Add some "material redistribution" by smoothing unconstrained regions
            self._redistribute_material()
    
    def _redistribute_material(self):
        """Redistribute material by local averaging in unconstrained regions."""
        # Create a mask for vertices that are neither boundary nor part of constrained edges
        constrained_vertices = set()
        for edge_idx in self.constrained_edges:
            constrained_vertices.update(self.edges[edge_idx])
        
        free_vertices = np.ones(len(self.vertices), dtype=bool)
        free_vertices[list(constrained_vertices)] = False
        free_vertices[self.is_boundary] = False
        
        if not np.any(free_vertices):
            return
        
        # Create adjacency list for quick neighbor lookup
        adjacency = [[] for _ in range(len(self.vertices))]
        for v1, v2 in self.edges:
            adjacency[v1].append(v2)
            adjacency[v2].append(v1)
        
        # Compute averaged positions for free vertices
        new_positions = self.vertices.copy()
        for idx in np.where(free_vertices)[0]:
            neighbors = adjacency[idx]
            if neighbors:
                # Weight based on inverse strain of connecting edges
                weights = []
                neighbor_positions = []
                for neighbor in neighbors:
                    edge_idx = np.where((self.edges == idx) & (self.edges == neighbor))[0]
                    if len(edge_idx) == 0:
                        edge_idx = np.where((self.edges == neighbor) & (self.edges == idx))[0]
                    
                    if len(edge_idx) > 0:
                        edge_vector = self.vertices[neighbor] - self.vertices[idx]
                        current_length = np.linalg.norm(edge_vector)
                        strain = abs(current_length - self.edge_rest_lengths[edge_idx[0]]) / self.edge_rest_lengths[edge_idx[0]]
                        weight = 1.0 / (1.0 + strain)
                        weights.append(weight)
                        neighbor_positions.append(self.vertices[neighbor])
                
                if weights:
                    weights = np.array(weights)
                    weights /= np.sum(weights)
                    new_positions[idx] = np.average(neighbor_positions, axis=0, weights=weights)
        
        # Apply smoothed positions with relaxation factor
        relaxation = 0.5
        self.vertices[free_vertices] = (1 - relaxation) * self.vertices[free_vertices] + \
                                     relaxation * new_positions[free_vertices]

    def simulate(self, num_iterations=100):
        """Run the simulation with improved stability and constraint satisfaction."""
        # Gradually increase stiffness for better convergence
        initial_stiffness = self.constraint_stiffness
        for iteration in range(num_iterations):
            print(iteration)
            # Adaptively adjust stiffness
            progress = iteration / num_iterations
            self.constraint_stiffness = initial_stiffness * (1.0 + 2.0 * progress)
            
            # Store previous positions
            prev_positions = self.vertices.copy()
            
            # Apply constraints with sub-iterations
            self._apply_edge_constraints()
            
            # Update velocities with adaptive damping
            non_boundary = ~self.is_boundary
            self.velocities[non_boundary] = (
                (self.vertices[non_boundary] - prev_positions[non_boundary]) / self.dt
            )
            
            # Adaptive damping based on progress
            adaptive_damping = self.damping * (1.0 - 0.5 * progress)
            self.velocities *= adaptive_damping
            
            # Apply velocity updates
            self.vertices[non_boundary] += self.velocities[non_boundary] * self.dt
    
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
        
        plt.show()

# Example usage
if __name__ == "__main__":
    NROWS = 10
    NCOLS = 10
    NEDGES = (NROWS - 1) * NCOLS * 2
    
    # Create simulator with a 10x10 grid
    simulator = GridClothSimulator(rows=NROWS, cols=NCOLS, spacing=1.0)
    
    edge_idxs = random.sample(range(0, NEDGES), NEDGES // 5)
    
    edge_idxs = [12, 13, 14, 15]
    
    for edge_idx in edge_idxs:
        # Add a constraint to make one edge shorter
        simulator.add_edge_constraint(edge_idx, 0.6)
    
    # Visualize initial state
    simulator.visualize("Initial State")
    
    # Run simulation
    simulator.simulate(num_iterations=15)
    
    # Visualize final state
    simulator.visualize("Final State")