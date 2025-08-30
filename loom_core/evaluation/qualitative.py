from typing import List, Dict, Tuple
from copy import deepcopy
import os
import numpy as np
import cv2

from loom.eval.vis import (
    QualitativeMesh,
    PatchProcessor,
    mesh_to_image,
    draw_mesh_on_canvas
)
from loom.eval.const import MESH_OFFSETS_DICT


def combine_meshes_on_canvas(meshes, canvas_size, img_size, 
                           front_color=(0, 255, 255), 
                           back_color=(0, 165, 255)):
    canvas = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)

    for mesh_obj in meshes:
        mesh_image = mesh_to_image(mesh_obj.mesh, image_size=img_size)
        color = front_color if mesh_obj.is_front else back_color
        draw_mesh_on_canvas(canvas, mesh_image, mesh_obj.offset, color)

    return canvas

    
def process_2d_meshes(
        patches: List[QualitativeMesh], 
        offsets: Dict[str, Tuple[float, float]],
        output_rootdir: str,
        experiment_name: str
    ):
    processor = PatchProcessor(patches, offsets)
    
    # Process meshes
    processor.apply_offsets()
    processor.combine_meshes()
    
    for subdir in [os.path.join(f'{output_rootdir}/{file_format}/{experiment_name}/') for file_format in ['ply', 'dxf', 'pdf', 'svg']]:
        os.makedirs(subdir, exist_ok=True)
    
    # Export to different formats
    processor.export_ply(f"{output_rootdir}/ply/{experiment_name}/pattern.ply")
    processor.export_dxf_variants(f"{output_rootdir}/dxf/{experiment_name}/pattern.dxf")
    processor.export_pdf_variants(f"{output_rootdir}/pdf/{experiment_name}/pattern.pdf")
    processor.export_svg_variants(f"{output_rootdir}/svg/{experiment_name}/pattern.svg")
        

def qualitative_evaluation(experiment_name):
    meshes_dict = {}
    param_2d_dir = 'results/pattern/latest/'
    output_rootdir = 'results/qualitative/pattern/'
    for _, patch_labels, _ in os.walk(param_2d_dir):
        for patch_label in patch_labels:
            if not(('upper' in patch_label or 'sleeve' in patch_label) and 'subject' in experiment_name):
                for param_2d_fname in os.listdir(os.path.join(param_2d_dir, patch_label)):
                    if 'optim' in param_2d_fname and 'ply' in param_2d_fname:
                        suffix = param_2d_fname.split('.')[0][6:]
                        qualitative_mesh = QualitativeMesh(patch_label, param_2d_fname)
                        if suffix not in meshes_dict:
                            meshes_dict[suffix] = [qualitative_mesh]
                        else:
                            meshes_dict[suffix].append(qualitative_mesh)
    
    output_dir_method = os.path.join(output_rootdir, 'png', experiment_name)  
    os.makedirs(output_dir_method, exist_ok=True)     

    final_meshes = deepcopy(meshes_dict['final-seams'])
    
    for suffix in meshes_dict:
        meshes_dict[suffix].sort(key=lambda mesh: mesh.is_front)

        canvas_size = (1200, 1200)
        img_size = (450, 450)

        patch_canvas = combine_meshes_on_canvas(meshes_dict[suffix], canvas_size, img_size)

        cv2.imwrite(os.path.join(output_dir_latest, 'png', f'sewing_pattern_{experiment_name}_{suffix}.png'), patch_canvas)
        cv2.imwrite(os.path.join(output_dir_method, f'sewing_pattern_{suffix}.png'), patch_canvas)
    
    process_2d_meshes(
        patches=final_meshes, 
        offsets=MESH_OFFSETS_DICT, 
        output_rootdir=output_rootdir,
        experiment_name=experiment_name
    )
