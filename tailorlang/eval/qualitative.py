import ezdxf
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import cv2
import os

from tailorlang.eval.patch_utils import (
    combine_meshes_on_canvas,
    Mesh,
    PatchProcessor,
    IMG_OFFSETS_DICT,
    MESH_OFFSETS_DICT
)

    
def process_2d_meshes(
        patches: List[Mesh], 
        offsets: Dict[str, Tuple[float, float]],
        output_dir: str,
        method: str
    ):
    """
    Process garment meshes and export to multiple formats.
    
    Args:
        patches: List of GarmentPatch objects
        offsets: Dictionary of offsets for each patch
        output_dir: Directory for output files
        pdf_page_size: Page size for PDF export ('A0' or 'A1')
    """
    processor = PatchProcessor(patches, offsets)
    
    # Process meshes
    processor.apply_offsets()
    processor.combine_meshes()
    
    # Export to different formats
    processor.export_ply(f"{output_dir}/pattern_{method}.ply")
    processor.export_dxf(f"{output_dir}/pattern_{method}.dxf")
        

def visualize_pattern(method):
    meshes_dict = {}
    data_dir = 'data/param_2d/'
    output_rootdir = 'results/qualitative/pattern/'
    for _, dirs, _ in os.walk(data_dir):
        for subdir in dirs:
            for fname in os.listdir(os.path.join(data_dir, subdir)):
                if 'optim' in fname and 'ply' in fname:
                    suffix = fname.split('.')[0][6:]
                    pattern_fpath = os.path.join(data_dir, subdir, fname)
                    is_front = True if subdir.split('_')[1] == 'front' else False
                    if suffix not in meshes_dict:
                        meshes_dict[suffix] = [
                            Mesh(pattern_fpath, IMG_OFFSETS_DICT[subdir], subdir, is_front)]
                    else:
                        meshes_dict[suffix].append(
                            Mesh(pattern_fpath, IMG_OFFSETS_DICT[subdir], subdir, is_front))
    
    output_dir_latest = os.path.join(output_rootdir, 'latest/')
    output_dir_method = os.path.join(output_rootdir, method)
    os.makedirs(output_dir_latest, exist_ok=True)     
    os.makedirs(output_dir_method, exist_ok=True)     
    
    final_meshes = deepcopy(meshes_dict['final-seams'])
    
    for suffix in meshes_dict:
        meshes_dict[suffix].sort(key=lambda mesh: mesh.is_front)

        canvas_size = (1200, 1200)
        img_size = (450, 450)
        combined_image = combine_meshes_on_canvas(meshes_dict[suffix], canvas_size, img_size)

        cv2.imwrite(
            os.path.join(output_dir_latest, 'png', f'sewing_pattern_{method}_{suffix}.png'), 
            combined_image
        )
        cv2.imwrite(
            os.path.join(output_dir_method, 'png', f'sewing_pattern_{method}_{suffix}.png'),
            combined_image
        )
    
    output_dir = os.path.join(output_rootdir, 'dfx', method)
    os.makedirs(output_dir, exist_ok=True)
    process_2d_meshes(
        patches=final_meshes, 
        offsets=MESH_OFFSETS_DICT, 
        output_dir=output_dir,
        method=method
    )


if __name__ == '__main__':
    visualize_pattern('average_50.0_10i')
