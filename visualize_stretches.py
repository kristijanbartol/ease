from typing import Tuple
import numpy as np
import trimesh
import os
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pyrender


def color_code_mesh(
        vertices: np.ndarray,   # N x 3
        faces: np.ndarray,      # N x 3
        stretches: np.ndarray,  # N x 3
        global_norm: Normalize,
        patch_norm: Normalize,
        colormap: cm.viridis,
        out_path: str
    ) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    global_norm_coefs = global_norm(stretches)
    patch_norm_coefs = patch_norm(stretches)

    global_norm_colors = (colormap(global_norm_coefs)[:, :3] * 255).astype(np.uint8)
    patch_norm_colors = (colormap(patch_norm_coefs)[:, :3] * 255).astype(np.uint8)
    
    global_norm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=global_norm_colors)
    patch_norm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=patch_norm_colors)
    global_norm_mesh.export(f'{out_path.split(".")[0]}_global_norm.ply')
    patch_norm_mesh.export(f'{out_path.split(".")[0]}_patch_norm.ply')

    return global_norm_mesh, patch_norm_mesh


def visualize_color_scale(norm, colormap, name):
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=ax, orientation='horizontal')
    cbar.set_label('Normalized Stretch Values')
    
    plt.title(name)
    plt.show()


def visualize_mesh(mesh, name):
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True, viewer_flags={'window_title': name})


def load_stretches(filename: str) -> np.ndarray:
    with open(filename, 'r') as file:
        lines = file.readlines()
        numbers = [float(line.strip()) for line in lines]
    return np.array(numbers)


def check_fname(patch_dir, stretch_fname):
    # For sleeves, takes *v* suffixes and for others take *u* suffixes
    if ('sleeve' in patch_dir and ('_v_' in stretch_fname or '_v.' in stretch_fname)) or \
            ('sleeve' not in patch_dir and ('_u_' in stretch_fname or '_u.' in stretch_fname)):
        return True
    else:
        return False
    

def extract_global_extremes(data_dir):
    global_max = -987654321
    global_min =  987654321
    for patch in os.listdir(data_dir):
        patch_dir = os.path.join(data_dir, patch)

        if os.path.isdir(patch_dir):
            stretch_fnames = [x for x in os.listdir(patch_dir) if 'stretches' in x and '.txt' in x]
            all_patch_stretches = []

            for stretch_fname in stretch_fnames:
                if check_fname(patch_dir=patch_dir, stretch_fname=stretch_fname):
                    stretch_path = os.path.join(patch_dir, stretch_fname)
                    all_patch_stretches.append(load_stretches(stretch_path))
            
            patch_stretches_array = np.stack(all_patch_stretches)
            patch_min = np.min(patch_stretches_array)
            patch_max = np.max(patch_stretches_array)

            if patch_min < global_min:
                global_min = patch_min
            if patch_max > global_max:
                global_max = patch_max
    
    return global_min, global_max


def get_patch_stretches_array(patch_dir):
    stretch_fnames = [x for x in os.listdir(patch_dir) if 'stretches' in x and '.txt' in x]
    all_patch_stretches = []
    for stretch_fname in stretch_fnames:
        if check_fname(patch_dir=patch_dir, stretch_fname=stretch_fname):
            stretch_path = os.path.join(patch_dir, stretch_fname)
            all_patch_stretches.append(load_stretches(stretch_path))
    
    return np.stack(all_patch_stretches), stretch_fnames


DATA_DIR = 'data/embedded/latest/skintight/'
COLORMAP = cm.viridis


if __name__ == '__main__':
    all_colored_meshes = []
    global_min, global_max = extract_global_extremes(DATA_DIR)
    global_norm = Normalize(vmin=global_min, vmax=global_max)
    visualize_color_scale(global_norm, COLORMAP, 'global')

    relevant_item_names = []
    for patch in os.listdir(DATA_DIR):
        patch_dir = os.path.join(DATA_DIR, patch)
        if os.path.isdir(patch_dir):
            patch_stretches_array, stretch_fnames = get_patch_stretches_array(patch_dir)
            patch_stretch_max = np.max(patch_stretches_array)
            patch_stretch_min = np.min(patch_stretches_array)
            patch_norm = Normalize(vmin=patch_stretch_min, vmax=patch_stretch_max)
            if 'upper_front' in patch_dir:
                visualize_color_scale(patch_norm, COLORMAP, f'{patch_dir.split("/")[-1]}')

            colored_meshes = []
            fidx = 0
            for stretch_fname in stretch_fnames:
                if check_fname(patch_dir=patch_dir, stretch_fname=stretch_fname):
                    stretch_path = os.path.join(patch_dir, stretch_fname)
                    mesh = trimesh.load(os.path.join(patch_dir, 'init.ply'))
                    global_norm_mesh, patch_norm_mesh = color_code_mesh(
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        stretches=patch_stretches_array[fidx],
                        global_norm=global_norm,
                        patch_norm=patch_norm,
                        colormap=COLORMAP,
                        out_path=os.path.join(patch_dir, f'{stretch_fname.split(".")[0]}.ply')
                    )
                    colored_meshes.append(global_norm_mesh)
                    if 'upper_front' in patch_dir:
                        item_name = f'{patch_dir.split("/")[-1]}-{stretch_fname.split("/")[-1].split(".")[0]}'
                        relevant_item_names.append(item_name)
                        visualize_mesh(patch_norm_mesh, item_name)
                    fidx += 1
            
            all_colored_meshes.append(np.stack(colored_meshes))
    
    # NOTE: Using a global norm makes it less convenient to see individual details
    # NOTE: I suggest using the individual patch norms as a reference (not the complete body)
    for stretch_item_idx in range(all_colored_meshes[0].shape[0]):
        scene = pyrender.Scene()
        for patch_idx in range(len(all_colored_meshes)):
            mesh = pyrender.Mesh.from_trimesh(all_colored_meshes[patch_idx][stretch_item_idx], smooth=False)
            scene.add(mesh)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, 
                                 viewer_flags={'window_title': f'{relevant_item_names[stretch_item_idx]}'})
