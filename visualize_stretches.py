import numpy as np
import trimesh
import os
from matplotlib import cm
from matplotlib.colors import Normalize


def color_code_mesh(
        vertices: np.ndarray,   # N x 3
        faces: np.ndarray,      # N x 3
        stretches: np.ndarray,  # N x 3
        stretch_min: float,     # scalar
        stretch_max: float,     # scalar
        out_path: str
    ) -> None:
    norm = Normalize(vmin=stretch_min, vmax=stretch_max)
    normalized_coefficients = norm(stretches)
    
    colormap = cm.viridis
    colors = (colormap(normalized_coefficients)[:, :3] * 255).astype(np.uint8)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=colors)
    mesh.export(out_path)


def load_stretches(filename: str) -> np.ndarray:
    with open(filename, 'r') as file:
        lines = file.readlines()
        numbers = [float(line.strip()) for line in lines]
    return np.array(numbers)


def check_fname(patch_dir, stretch_fname):
    if ('sleeve' in patch_dir and ('_v_' in stretch_fname or '_v.' in stretch_fname)) or \
            ('sleeve' not in patch_dir and ('_u_' in stretch_fname or '_u.' in stretch_fname)):
        return True
    else:
        return False


DATA_DIR = 'data/embedded/latest/skintight/'


if __name__ == '__main__':
    for patch in os.listdir(DATA_DIR):
        patch_dir = os.path.join(DATA_DIR, patch)
        if os.path.isdir(patch_dir):
            stretch_fnames = [x for x in os.listdir(patch_dir) if 'stretches' in x and '.txt' in x]
            all_patch_stretches = []
            for stretch_fname in stretch_fnames:
                if check_fname(patch_dir=patch_dir, stretch_fname=stretch_fname):
                    stretch_path = os.path.join(patch_dir, stretch_fname)
                    all_patch_stretches.append(load_stretches(stretch_path))
            
            patch_stretches_array = np.stack(all_patch_stretches)
            patch_stretch_max = np.max(all_patch_stretches)
            patch_stretch_min = np.min(all_patch_stretches)

            fidx = 0
            for stretch_fname in stretch_fnames:
                if check_fname(patch_dir=patch_dir, stretch_fname=stretch_fname):
                    stretch_path = os.path.join(patch_dir, stretch_fname)
                    mesh = trimesh.load(os.path.join(patch_dir, 'init.ply'))
                    color_code_mesh(
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        stretches=patch_stretches_array[fidx],
                        stretch_min=patch_stretch_min,
                        stretch_max=patch_stretch_max,
                        out_path=os.path.join(patch_dir, f'{stretch_fname.split(".")[0]}.ply')
                    )
                    fidx += 1
