import trimesh
import numpy as np
import pyrender
import torch
from smplx import SMPL


def check_inversion(mesh, vertex_indices):
    """
    Check if the deflation is causing an inversion at the vertices specified by vertex_indices.

    Parameters:
        mesh (trimesh.Trimesh): The mesh to check.
        vertex_indices (np.array): Indices of vertices to check.

    Returns:
        np.array: A boolean array indicating if the corresponding vertex is causing inversion.
    """
    inverted = np.zeros(len(vertex_indices), dtype=bool)
    for i, index in enumerate(vertex_indices):
        # Get the adjacent faces to the vertex
        adjacent_faces = mesh.vertex_faces[index]
        face_normals = mesh.face_normals[adjacent_faces]
        
        # Check the angle between normals of each pair of adjacent faces
        for j in range(len(face_normals)):
            for k in range(j+1, len(face_normals)):
                angle = np.arccos(np.clip(np.dot(face_normals[j], face_normals[k]), -1.0, 1.0))
                if np.degrees(angle) > 90:  # Inversion occurs if the angle exceeds 90 degrees
                    inverted[i] = True
                    break
            if inverted[i]:
                break
    return inverted

def deflate_mesh(mesh, step_size, iterations):
    """
    Deflates the given mesh by moving vertices inward along their normals,
    stopping if they reach a certain inward distance or if inversion is detected.

    Parameters:
        mesh (trimesh.Trimesh): The mesh to deflate.
        step_size (float): The magnitude of each inward step.
        iterations (int): Number of times to apply the deflation step.

    Returns:
        trimesh.Trimesh: The deflated mesh.
    """
    for _ in range(iterations):
        # Calculate the proposed new positions
        proposed_positions = mesh.vertices - step_size * mesh.vertex_normals

        # Check for inversion at each vertex
        inverted = check_inversion(mesh, np.arange(len(mesh.vertices)))

        # Apply the movement only to vertices that are not inverting the mesh
        mesh.vertices[~inverted] = proposed_positions[~inverted]

    return mesh

def main():
    # Load the SMPL mesh from a file
    smpl_model = SMPL(model_path='/home/kristijan/data/hierprob3d/smpl/SMPL_FEMALE.pkl')
    betas = torch.ones((1, 10)) * 7.5
    betas[0, 0] = 0.
    verts = smpl_model(betas=betas).vertices[0].cpu().detach().numpy()
    smpl_mesh = trimesh.Trimesh(vertices=verts, faces=smpl_model.faces)

    # Set the deflation parameters
    step_size = 0.0001  # Define how much each vertex should move inward per iteration
    iterations = 200   # Define how many times to apply the deflation

    # Deflate the mesh
    #deflated_mesh = deflate_mesh(smpl_mesh, step_size, iterations)
    #deflated_mesh = smpl_mesh

    # Visualize the original and deflated mesh using pyrender
    scene = pyrender.Scene()
    original_mesh_node = pyrender.Mesh.from_trimesh(smpl_mesh)
    #deflated_mesh_node = pyrender.Mesh.from_trimesh(deflated_mesh)
    
    scene.add(original_mesh_node)
    #scene.add(deflated_mesh_node)
    
    pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == '__main__':
    main()


# CONCLUSION: I WILL USE THE SIMULATION DIRECTLY IN HOUDINI, NO DEFLATING AND INFLATING WILL BE USED.
