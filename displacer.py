'''
import numpy as np
import trimesh

def scale_mesh_3d(mesh, edge_factors):
    """
    Scales the edges of the mesh in 3D space by given factors.

    :param mesh: The trimesh mesh instance.
    :param edge_factors: A list or array of scale factors for each edge.
    :return: Scaled mesh.
    """
    # Create a copy of the vertices to manipulate
    vertices = np.array(mesh.vertices)
    # Iterate over each edge and scale
    for i, edge in enumerate(mesh.edges_unique):
        # Scale factor for this edge
        scale_factor = edge_factors[i % len(edge_factors)]
        # Get the vertex positions for the edge
        v0, v1 = vertices[edge[0]], vertices[edge[1]]
        # Calculate the midpoint
        midpoint = (v0 + v1) / 2.0
        # Calculate the direction vector for the edge
        direction = v1 - v0
        # Scale the edge vertices in 3D
        vertices[edge[0]] = midpoint - direction * scale_factor / 2.0
        vertices[edge[1]] = midpoint + direction * scale_factor / 2.0
    # Update the mesh with the new vertices
    mesh.vertices = vertices
    return mesh

if __name__ == '__main__':
    # Load the mesh
    mesh = trimesh.load_mesh('path_to_your_mesh.ply')

    # Edge scaling factors, with random variation between 1.05 and 1.15
    random_factors = 1.05 + np.random.rand(len(mesh.edges_unique)) * 0.1

    # Scale the mesh edges in 3D
    scaled_mesh = scale_mesh_3d(mesh, random_factors)

    # Save the modified mesh
    scaled_mesh.export('scaled_mesh.ply')
'''

'''
import numpy as np
import trimesh


if __name__ == '__main__':
    # Load the mesh
    mesh = trimesh.load_mesh('output/orig_upper_garment.ply')

    # Create a scaling matrix
    scale_factor = 1.2
    scaling_matrix = np.array([
        [scale_factor, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, scale_factor, 0],  # Z-axis is not scaled
        [0, 0, 0, 1]
    ])

    # Apply the scaling
    mesh.apply_transform(scaling_matrix)

    # Save the modified mesh
    mesh.export('modified_mesh.ply')
'''

'''
import trimesh
import numpy as np

# Load the mesh
mesh = trimesh.load('output/orig_upper_garment.ply')

# Get edges from the mesh
edges = mesh.edges

# Get unique vertices in the edges
unique_vertices = np.unique(edges)

# Create a mapping for unique vertex indices to their new positions after scaling
vertex_map = {}

# For each vertex, compute a random scaling factor and apply it
for vertex_index in unique_vertices:
    # Random scaling factors between 1.15 and 1.25 for the X and Y directions
    scale_factor_x = np.random.uniform(1.15, 1.25)
    scale_factor_y = np.random.uniform(1.15, 1.25)

    # Get the original vertex position
    original_vertex = mesh.vertices[vertex_index]

    # Apply the scaling to the vertex position
    # Note: We're not scaling the Z position
    scaled_vertex = original_vertex * [scale_factor_x, 1, scale_factor_y]

    # Store the scaled vertex position in the mapping
    vertex_map[vertex_index] = scaled_vertex

# Update the vertices in the mesh with the new scaled positions
for vertex_index, new_position in vertex_map.items():
    mesh.vertices[vertex_index] = new_position

mesh.export('modified_mesh.ply')
'''



'''
import numpy as np
import trimesh
import torch
from scipy.optimize import minimize

# Example mesh loading (we will replace this with the actual mesh file later)
# mesh = trimesh.load('path_to_your_mesh_file')

# For the purpose of this demonstration, we will create a random mesh.
# This is a placeholder. In practice, you would load an actual mesh.
mesh = trimesh.creation.icosphere(subdivisions=2)

def objective_function(vertex_positions, mesh, target_lengths):
    mesh.vertices = vertex_positions.reshape(-1, 3)
    current_lengths = mesh.edges_unique_length
    length_differences = current_lengths - target_lengths
    return np.sum(length_differences ** 2)


def edge_constraints(vertex_positions, mesh, target_lengths):
    mesh.vertices = vertex_positions.reshape(-1, 3)
    current_lengths = mesh.edges_unique_length
    return current_lengths - target_lengths
'''
    

'''
def barycentric_coords(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(dim=1)
    d01 = (v0 * v1).sum(dim=1)
    d11 = (v1 * v1).sum(dim=1)
    d20 = (v2 * v0).sum(dim=1)
    d21 = (v2 * v1).sum(dim=1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=1)

def process_all_faces(V_2d, V_3d):
    num_faces = V_2d.shape[0]
    # Calculate centroid of all triangles in 2D
    D = torch.mean(V_2d, dim=1)
    D_bary = torch.full((num_faces, 3), 1/3, device=V_2d.device)

    # Displaced points in the UV coordinate plane
    DU = D.clone()
    DU[:, 0] += 1.0
    DV = D.clone()
    DV[:, 1] += 1.0

    # Barycentric coordinates for all displaced points
    DU_bary = barycentric_coords(DU, V_2d[:, 0], V_2d[:, 1], V_2d[:, 2])
    DV_bary = barycentric_coords(DV, V_2d[:, 0], V_2d[:, 1], V_2d[:, 2])

    # Project these points into 3D using the barycentric coordinates
    Dp = (D_bary[:, :, None] * V_3d).sum(dim=1)
    DUp = (DU_bary[:, :, None] * V_3d).sum(dim=1)
    DVp = (DV_bary[:, :, None] * V_3d).sum(dim=1)

    # Calculate stretch factors in the u and v directions for all faces
    target_u = torch.norm(DUp - Dp, dim=1) * 1.2
    target_v = torch.norm(DVp - Dp, dim=1)

    return torch.stack([target_u, target_v], dim=1)


if __name__ == '__main__':
    # Generate random scaling factors for each edge
    scaling_factors = np.random.uniform(1.15, 1.25, size=mesh.edges_unique_length.shape)

    # Compute the target lengths for each edge
    target_lengths = mesh.edges_unique_length * scaling_factors

    # Get the initial positions of the vertices (flattened array)
    initial_vertex_positions = mesh.vertices.flatten()

    # Set up bounds for the vertex positions if necessary (None means unbounded)
    bounds = [(None, None)] * len(initial_vertex_positions)

    # Perform the optimization
    result = minimize(
        fun=objective_function,
        x0=initial_vertex_positions,
        args=(mesh, target_lengths),
        bounds=bounds,
        constraints={'type': 'eq', 'fun': edge_constraints, 'args': (mesh, target_lengths)}
    )

    # Check if the optimization was successful
    if result.success:
        # Reshape the result into the vertex array
        optimized_vertex_positions = result.x.reshape((-1, 3))
        
        # Update the mesh with the optimized vertex positions
        mesh.vertices = optimized_vertex_positions

        # Visualize the optimized mesh
        mesh.show()
    else:
        print("Optimization failed:", result.message)
'''



'''
def objective_function(vertices, triangles, initial_areas, target_stretches):
    current_stretches = []
    for idx, triangle in enumerate(triangles):
        v0, v1, v2 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        current_area = calculate_triangle_area(v0, v1, v2)
        initial_area = initial_areas[idx]
        current_stretch = current_area / initial_area
        current_stretches.append(current_stretch)
    
    stretch_differences = torch.stack(current_stretches) - target_stretches  # Use torch.stack instead of torch.tensor
    objective_value = torch.sum(stretch_differences ** 2)
    return objective_value


def compute_cotangent_weights(vertices, triangles):
    num_vertices = vertices.size(0)
    adjacency_matrix = torch.zeros((num_vertices, num_vertices), dtype=torch.float32)
    degree_matrix = torch.zeros((num_vertices, num_vertices), dtype=torch.float32)
    
    for tri in triangles:
        i, j, k = tri[0], tri[1], tri[2]
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        
        # Vector from k to i and k to j
        vki = vi - vk
        vkj = vj - vk
        
        # Compute cotangent of the angle at each vertex
        cot_ki_j = torch.dot(vki, vkj) / torch.norm(torch.cross(vki, vkj))
        
        # Symmetric updates for the adjacency matrix
        adjacency_matrix[i, j] += cot_ki_j
        adjacency_matrix[j, i] += cot_ki_j
        
        # Update degree matrix
        degree_matrix[i, i] += cot_ki_j
        degree_matrix[j, j] += cot_ki_j

    return adjacency_matrix, degree_matrix

def calculate_triangle_area(v0, v1, v2):
    return torch.linalg.norm(torch.cross(v1 - v0, v2 - v0)) / 2
    

def calculate_triangle_area_from_index(vertices, triangle_indices):
    v0, v1, v2 = vertices[triangle_indices[0]], vertices[triangle_indices[1]], vertices[triangle_indices[2]]
    return torch.linalg.norm(torch.cross(v1 - v0, v2 - v0)) / 2
'''


import numpy as np
import torch
import trimesh


def calculate_triangle_areas(vertices, triangles):
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    area = torch.linalg.norm(torch.cross(v1 - v0, v2 - v0), dim=1) / 2
    return area


def calculate_target_stretches(parameterized_vertices, triangles, initial_areas):
    parameterized_areas = calculate_triangle_areas(parameterized_vertices, triangles)
    target_stretches = parameterized_areas / initial_areas
    return target_stretches
    

def objective_function_parallel(vertices, triangles, initial_areas, target_stretches):
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    areas = torch.linalg.norm(torch.cross(v1 - v0, v2 - v0), dim=1) / 2
    current_stretches = areas / initial_areas
    stretch_differences = current_stretches - target_stretches

    return torch.sum(stretch_differences ** 2)


def compute_cotangent_weights_parallel(vertices, triangles):
    num_vertices = vertices.size(0)
    adjacency_matrix = torch.zeros((num_vertices, num_vertices), dtype=torch.float32)
    degree_matrix = torch.zeros((num_vertices, num_vertices), dtype=torch.float32)

    v_i = vertices[triangles[:, 0]]  # vertices at first corner
    v_j = vertices[triangles[:, 1]]  # vertices at second corner
    v_k = vertices[triangles[:, 2]]  # vertices at third corner

    vki = v_i - v_k  # Vector from vertex k to i
    vkj = v_j - v_k  # Vector from vertex k to j

    # Compute cotangents of angles between vectors
    dot_product = (vki * vkj).sum(-1)  # Dot product of vectors
    cross_norm = torch.linalg.norm(torch.cross(vki, vkj), dim=1)  # Norm of cross product
    cot_ki_j = dot_product / cross_norm  # Cotangent of angle

    # Update adjacency matrix
    i, j, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    adjacency_matrix[i, j] += cot_ki_j
    adjacency_matrix[j, i] += cot_ki_j

    # Update degree matrix
    degree_matrix[i, i] += cot_ki_j
    degree_matrix[j, j] += cot_ki_j

    return adjacency_matrix, degree_matrix


def compute_laplacian_loss(vertices, laplacian_matrix, original_laplacian_coords):
    current_laplacian_coords = torch.matmul(laplacian_matrix, vertices)
    laplacian_loss = torch.mean((current_laplacian_coords - original_laplacian_coords) ** 2)
    return laplacian_loss


def compute_laplacian_matrix(adjacency_matrix, degree_matrix):
    return degree_matrix - adjacency_matrix


def regularization_loss(vertices, original_vertices, gamma):
    return torch.sum((vertices - original_vertices) ** 2)


if __name__ == '__main__':
    mesh = trimesh.load('results/tl_out/orig_front_shirt_0.001.obj')
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, requires_grad=True) # type: ignore
    triangles = torch.tensor(mesh.faces, dtype=torch.int32) # type: ignore

    parameterized_mesh = trimesh.load('results/optim_in/param_front_shirt.obj')
    parameterized_vertices = torch.tensor(parameterized_mesh.vertices, dtype=torch.float32) # type: ignore

    orig_verts = vertices.clone().detach()

    num_iterations = 500
    alpha = 1.0
    beta = 1.
    gamma = 0.

    initial_areas = calculate_triangle_areas(orig_verts, triangles)
    adjacency_matrix, degree_matrix = compute_cotangent_weights_parallel(orig_verts, triangles)
    laplacian_matrix = compute_laplacian_matrix(adjacency_matrix, degree_matrix)
    original_laplacian_coords = torch.matmul(laplacian_matrix, orig_verts)
    target_stretches = calculate_target_stretches(parameterized_vertices, triangles, initial_areas)

    target_stretches = torch.ones_like(target_stretches) * 1.5

    optimizer = torch.optim.Adam([vertices], lr=0.0025)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        if iteration % 80 == 0:
            print('')

        stretch_loss = objective_function_parallel(vertices, triangles, initial_areas, target_stretches)
        laplacian_loss = compute_laplacian_loss(vertices, laplacian_matrix, original_laplacian_coords)
        reg_loss = regularization_loss(vertices, orig_verts, gamma)
        
        total_loss = alpha * stretch_loss + beta * laplacian_loss + gamma * reg_loss

        total_loss.backward()
        optimizer.step()

        if iteration % 5 == 0:
            print(f"Iteration {iteration}, Loss: {total_loss.item():.5f} (Stretch: {stretch_loss:.5f}, Laplacian: {laplacian_loss:.5f}, Reg: {reg_loss:.5f})")
        
        if total_loss < 0.01:
            break

    optimized_mesh = trimesh.Trimesh(vertices=vertices.detach().numpy(), faces=triangles.numpy())
    optimized_mesh.export('results/optim_out/optimized_fron_shirt.ply')
