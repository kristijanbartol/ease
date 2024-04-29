import numpy as np
import pandas as pd
import trimesh


def calculate_warp_stretch(initial_vertices, target_vertices, faces, warp_direction):
    warp_stretches = []
    
    for face in faces:
        # Extract the vertices for each triangle
        init_v0, init_v1, init_v2 = initial_vertices[face]
        targ_v0, targ_v1, targ_v2 = target_vertices[face]

        # Calculate edge vectors in initial and target mesh
        init_edge1 = init_v1 - init_v0
        init_edge2 = init_v2 - init_v0
        targ_edge1 = targ_v1 - targ_v0
        targ_edge2 = targ_v2 - targ_v0
        
        # Project edge vectors onto the warp direction
        init_proj1 = np.dot(init_edge1, warp_direction)
        init_proj2 = np.dot(init_edge2, warp_direction)
        targ_proj1 = np.dot(targ_edge1, warp_direction)
        targ_proj2 = np.dot(targ_edge2, warp_direction)
        
        # Calculate the magnitude of the projection on the warp direction
        init_magnitude = np.sqrt(init_proj1**2 + init_proj2**2)
        targ_magnitude = np.sqrt(targ_proj1**2 + targ_proj2**2)
        
        # Calculate the stretch (target / initial)
        if init_magnitude != 0:  # Avoid division by zero
            stretch = targ_magnitude / init_magnitude
        else:
            stretch = 0  # Default value if initial magnitude is zero
        warp_stretches.append(stretch)
    
    return warp_stretches


if __name__ == '__main__':
    warp_direction = np.array([1, 0, 0], dtype=np.float32)
    garment_mesh = trimesh.load('results/optim_in/orig_front_shirt_0.001.ply')
    target_mesh = trimesh.load('results/optim_in/param_front_shirt_0.001.obj')

    garment_verts = garment_mesh.vertices
    target_verts = target_mesh.vertices
    faces = garment_mesh.faces

    target_stretches = calculate_warp_stretch(
        initial_vertices=garment_verts,
        target_vertices=target_verts,
        faces=faces,
        warp_direction=warp_direction
    )

    df = pd.DataFrame({
        'target_stretch': target_stretches
    })

    csv_file_path = 'target_stretches.csv'
    df.to_csv(csv_file_path, index=False)

    print(f"CSV file has been created at {csv_file_path}")
