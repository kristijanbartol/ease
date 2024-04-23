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
'''

import os
import numpy as np
import trimesh
import torch
from scipy.spatial import KDTree


def compute_jacobians(vertices, triangles, warp_direction, weft_direction):
    # Extract vertices of the triangles
    v0s = vertices[triangles[:, 0]]
    v1s = vertices[triangles[:, 1]]
    v2s = vertices[triangles[:, 2]]

    # Compute edge vectors for all triangles simultaneously
    edge1s = v1s - v0s
    edge2s = v2s - v0s

    # Project edge vectors onto warp and weft directions for all triangles
    J_warp = torch.stack((torch.sum(edge1s * warp_direction, dim=1),
                          torch.sum(edge2s * warp_direction, dim=1)), dim=1)
    J_weft = torch.stack((torch.sum(edge1s * weft_direction, dim=1),
                          torch.sum(edge2s * weft_direction, dim=1)), dim=1)

    # Form the Jacobian matrices from the projections
    J = torch.stack((J_warp, J_weft), dim=2)
    return J

def jacobian_loss_function(initial_jacobians, target_jacobians):
    # Compute the Frobenius norm of the difference between Jacobians for all triangles
    loss = torch.norm(initial_jacobians - target_jacobians, dim=(1, 2))
    return torch.mean(loss)


def compute_sdf(mesh, grid_resolution=100, device='cuda'):
    """
    Compute the signed-distance function for a given mesh using pure PyTorch (no Kaolin).

    Parameters:
    - mesh (trimesh.Trimesh): Trimesh object of the mesh
    - grid_resolution (int): The resolution of the grid in each dimension.
    - device (str): Device to perform computation ('cuda' or 'cpu')

    Returns:
    - torch.Tensor: The SDF grid values shaped in a 3D tensor.
    - torch.Tensor: The grid points as a single tensor of 3D coordinates.
    """
    # Define the bounds of the grid based on the mesh bounds
    bounds = mesh.bounds
    grid_min, grid_max = np.array(bounds[0]), np.array(bounds[1])

    # Create a grid using torch
    x = torch.linspace(grid_min[0], grid_max[0], grid_resolution, device=device)
    y = torch.linspace(grid_min[1], grid_max[1], grid_resolution, device=device)
    z = torch.linspace(grid_min[2], grid_max[2], grid_resolution, device=device)
    xv, yv, zv = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack((xv, yv, zv), dim=-1).reshape(-1, 3)

    # Mesh vertices and faces to tensors
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)

    # Compute distance from each grid point to the nearest mesh vertex
    dists = torch.cdist(grid_points, vertices)
    min_dists, _ = torch.min(dists, dim=1)
    sdf_grid = min_dists.reshape(grid_resolution, grid_resolution, grid_resolution)

    sdf_grid = sdf_grid.detach().cpu().numpy()
    grid_points = grid_points.detach().cpu().numpy()

    return sdf_grid, grid_points


def trilinear_interpolation(grid, points, grid_min, grid_max):
    """
    Perform trilinear interpolation on a 3D grid at specified points.
    
    Parameters:
    - grid (torch.Tensor): The 3D tensor representing the SDF grid.
    - points (torch.Tensor): Nx3 tensor of points where SDF values need to be interpolated.
    - grid_min (torch.Tensor): The minimum coordinate of the SDF grid in 3D space.
    - grid_max (torch.Tensor): The maximum coordinate of the SDF grid in 3D space.
    
    Returns:
    - torch.Tensor: Interpolated SDF values at the given points.
    """
    # Ensure that all tensors are on the same device
    device = grid.device
    points = points.to(device)
    grid_min = grid_min.to(device)
    grid_max = grid_max.to(device)

    # Normalize points to the grid scale
    grid_size = torch.tensor(grid.shape, device=device).float() - 1
    points_scaled = (points - grid_min) / (grid_max - grid_min) * grid_size

    # Compute the indices of the grid points to interpolate
    x0 = points_scaled[:, 0].floor().clamp(0, grid_size[0])
    x1 = (x0 + 1).clamp(0, grid_size[0])
    y0 = points_scaled[:, 1].floor().clamp(0, grid_size[1])
    y1 = (y0 + 1).clamp(0, grid_size[1])
    z0 = points_scaled[:, 2].floor().clamp(0, grid_size[2])
    z1 = (z0 + 1).clamp(0, grid_size[2])

    # Interpolation weights
    xd = (points_scaled[:, 0] - x0)
    yd = (points_scaled[:, 1] - y0)
    zd = (points_scaled[:, 2] - z0)

    # Fetch grid values at corner points using advanced indexing
    x0, x1, y0, y1, z0, z1 = [x.long() for x in [x0, x1, y0, y1, z0, z1]]
    c000 = grid[x0, y0, z0]
    c001 = grid[x0, y0, z1]
    c010 = grid[x0, y1, z0]
    c011 = grid[x0, y1, z1]
    c100 = grid[x1, y0, z0]
    c101 = grid[x1, y0, z1]
    c110 = grid[x1, y1, z0]
    c111 = grid[x1, y1, z1]

    # Trilinear interpolation
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd
    return c


ALPHA = 1.0
BETA = .0
N_ITER = 500
GRID_RES = 70


if __name__ == '__main__':
    # Load meshes, setup tensors, etc.
    body_mesh = trimesh.load('results/tl_out/orig_body.ply')

    garment_mesh = trimesh.load('results/optim_in/orig_front_shirt_0.001.ply')
    garment_vertices = torch.tensor(garment_mesh.vertices, dtype=torch.float32, requires_grad=True)
    garment_triangles = torch.tensor(garment_mesh.faces, dtype=torch.int32)

    parameterized_mesh = trimesh.load('results/optim_in/param_front_shirt_0.001.obj')
    parameterized_vertices = torch.tensor(parameterized_mesh.vertices, dtype=torch.float32)

    # Set parameters
    warp_direction = torch.tensor([1, 0, 0], dtype=torch.float32)
    weft_direction = torch.tensor([0, 1, 0], dtype=torch.float32)

    # Compute Jacobians for all triangles in both meshes
    initial_jacobians = compute_jacobians(garment_vertices, garment_triangles, warp_direction, weft_direction)
    target_jacobians = compute_jacobians(parameterized_vertices, garment_triangles, warp_direction, weft_direction)

    sdf_grid_path = f'results/optim_in/sdf_grid_{GRID_RES}.npy'
    grid_points_path = f'results/optim_in/grid_points_{GRID_RES}.npy'

    if not os.path.exists(sdf_grid_path):
        sdf_grid, grid_points = compute_sdf(body_mesh, grid_resolution=GRID_RES)
        #np.save(sdf_grid_path, sdf_grid)
        #np.save(grid_points_path, grid_points)
    else:
        sdf_grid = np.load(sdf_grid_path)
        grid_points = np.load(grid_points_path)

    sdf_grid_tensor = torch.tensor(sdf_grid, dtype=torch.float32)
    sdf_min = torch.tensor(grid_points.min(axis=0), dtype=torch.float32)#.to(device)
    sdf_max = torch.tensor(grid_points.max(axis=0), dtype=torch.float32)#.to(device)

    print('Calculated SDF and Jacobians, starting the optimization...')

    optimizer = torch.optim.Adam([garment_vertices], lr=0.001)

    for iteration in range(N_ITER):
        optimizer.zero_grad()
        
        # Compute Jacobians and SDF based loss
        current_jacobians = compute_jacobians(garment_vertices, garment_triangles, warp_direction, weft_direction)
        jacobian_loss = jacobian_loss_function(current_jacobians, target_jacobians)
        sdf_values = trilinear_interpolation(sdf_grid_tensor, garment_vertices, sdf_min, sdf_max)
        collision_loss = torch.mean(torch.relu(-sdf_values))  # Assuming SDF is negative inside the mesh

        total_loss = ALPHA * jacobian_loss + BETA * collision_loss
        
        total_loss.backward()
        optimizer.step()

        if iteration % 5 == 0:
            print(f"Iteration {iteration}, Total Loss: {total_loss.item()} (Jacobian: {jacobian_loss}, Collision: {collision_loss})")

        #if total_loss.item() < 0.001:
        #    print("Convergence reached")
        #    break

    final_vertices = garment_vertices.detach().cpu().numpy()
    optimized_mesh = trimesh.Trimesh(vertices=final_vertices, faces=garment_triangles.cpu().numpy())
    optimized_mesh.export('optimized_mesh.ply')


    '''
    # Load meshes, setup tensors, etc.
    mesh = trimesh.load('results/optim_in/orig_front_shirt_0.001.ply')
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, requires_grad=True)
    triangles = torch.tensor(mesh.faces, dtype=torch.int32)

    parameterized_mesh = trimesh.load('results/optim_in/param_front_shirt.obj')
    parameterized_vertices = torch.tensor(parameterized_mesh.vertices, dtype=torch.float32)

    # Define warp and weft directions (these should be normalized)
    warp_direction = torch.tensor([1, 0, 0], dtype=torch.float32)  # Example direction
    weft_direction = torch.tensor([0, 1, 0], dtype=torch.float32)

    # Compute Jacobians for all triangles in both meshes
    initial_jacobians = compute_jacobians(vertices, triangles, warp_direction, weft_direction)
    target_jacobians = compute_jacobians(parameterized_vertices, triangles, warp_direction, weft_direction)

    num_iterations = 100
    optimizer = torch.optim.Adam([vertices], lr=0.001)
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Recompute current Jacobians based on updated vertices
        current_jacobians = compute_jacobians(vertices, triangles, warp_direction, weft_direction)
        jacobian_loss = jacobian_loss_function(current_jacobians, target_jacobians)
        
        jacobian_loss.backward()
        optimizer.step()

        if iteration % 5 == 0:
            print(f"Iteration {iteration}, Jacobian Loss: {jacobian_loss.item()}")

    # Save or further process the optimized mesh
    optimized_mesh = trimesh.Trimesh(vertices=vertices.detach().numpy(), faces=triangles.numpy())
    optimized_mesh.export('optimized_mesh.ply')
    '''
    
