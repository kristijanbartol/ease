from typing import List, Dict, Tuple
from copy import deepcopy
import cv2
import os
import numpy as np

from eval.vis import (
    QualitativeMesh,
    PatchProcessor,
    mesh_to_image,
    draw_mesh_on_canvas
)
from tailorlang.eval.stretch_utils import (
    mesh_to_stretch_image
)
from tailorlang.eval.const import MESH_OFFSETS_DICT


'''
def combine_meshes_on_canvas(meshes, canvas_size, img_size, front_color=(0, 255, 255), back_color=(0, 165, 255)):
    # Create a blank canvas with alpha channel (transparent background)
    canvas = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)

    for mesh in meshes:
        mesh_image = mesh_to_image(mesh.mesh, image_size=img_size)
        color = front_color if mesh.is_front else back_color
        draw_mesh_on_canvas(canvas, mesh_image, mesh.offset, color)

    return canvas
'''

def combine_meshes_on_canvas(meshes, canvas_size, img_size, 
                           visualization_mode='sewing_pattern',
                           direction='weft',
                           front_color=(0, 255, 255), 
                           back_color=(0, 165, 255)):
    """
    Combine multiple mesh visualizations on a single canvas.
    
    Args:
        meshes: List of mesh objects with mesh, offset, is_front properties
        canvas_size: Size of the output canvas
        img_size: Size for individual mesh images
        visualization_mode: Either 'sewing_pattern' or 'stretch'
        direction: warp or weft
        front_color/back_color: Colors for front/back visualization mode
    """
    canvas = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)

    for mesh_obj in meshes:
        if visualization_mode == 'sewing_pattern':
            mesh_image = mesh_to_image(mesh_obj.mesh, image_size=img_size)
            color = front_color if mesh_obj.is_front else back_color
            draw_mesh_on_canvas(canvas, mesh_image, mesh_obj.offset, color)
        else:  # 'stretch' mode
            mesh_image = mesh_to_stretch_image(
                mesh_obj.mesh, 
                mesh_obj.face_scales_weft if direction == 'weft' else mesh_obj.face_scales_warp,
                image_size=img_size
            )
            draw_mesh_on_canvas(canvas, mesh_image, mesh_obj.offset, (255, 255, 255))

    return canvas

    
def process_2d_meshes(
        patches: List[QualitativeMesh], 
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
    processor.export_dxf_variants(f"{output_dir}/pattern_{method}.dxf")
    processor.export_pdf_variants(f"{output_dir}/pattern_{method}.pdf")
    processor.export_svg_variants(f"{output_dir}/pattern_{method}.svg")
        

def qualitative_evaluation(method):
    meshes_dict = {}
    param_2d_dir = 'data/param_2d/'
    output_rootdir = 'results/qualitative/pattern/'
    for _, patch_labels, _ in os.walk(param_2d_dir):
        for patch_label in patch_labels:
            for param_2d_fname in os.listdir(os.path.join(param_2d_dir, patch_label)):
                if 'optim' in param_2d_fname and 'ply' in param_2d_fname:
                    suffix = param_2d_fname.split('.')[0][6:]
                    qualitative_mesh = QualitativeMesh(patch_label, param_2d_fname)
                    if suffix not in meshes_dict:
                        meshes_dict[suffix] = [qualitative_mesh]
                    else:
                        meshes_dict[suffix].append(qualitative_mesh)
    
    output_dir_latest = os.path.join(output_rootdir, 'latest/')
    output_dir_method = os.path.join(output_rootdir, method)
    os.makedirs(os.path.join(output_dir_latest, 'png'), exist_ok=True)     
    os.makedirs(os.path.join(output_dir_method, 'png'), exist_ok=True)     
    
    final_meshes = deepcopy(meshes_dict['final-seams'])
    
    for suffix in meshes_dict:
        meshes_dict[suffix].sort(key=lambda mesh: mesh.is_front)

        canvas_size = (1200, 1200)
        img_size = (450, 450)
        
        # For stretch visualization
        stretch_canvas_weft = combine_meshes_on_canvas(
            meshes_dict[suffix], canvas_size, img_size, visualization_mode='stretch', direction='weft')
        stretch_canvas_warp = combine_meshes_on_canvas(
            meshes_dict[suffix], canvas_size, img_size, visualization_mode='stretch', direction='warp')

        # For original front/back visualization
        patch_canvas = combine_meshes_on_canvas(
            meshes_dict[suffix], canvas_size, img_size, visualization_mode='sewing_pattern')

        cv2.imwrite(
            os.path.join(output_dir_latest, 'png', f'weft_stretch_pattern_{method}_{suffix}.png'), stretch_canvas_weft)
        cv2.imwrite(
            os.path.join(output_dir_method, 'png', f'weft_stretch_pattern_{method}_{suffix}.png'), stretch_canvas_weft)
        cv2.imwrite(
            os.path.join(output_dir_latest, 'png', f'warp_stretch_pattern_{method}_{suffix}.png'), stretch_canvas_warp)
        cv2.imwrite(
            os.path.join(output_dir_method, 'png', f'warp_stretch_pattern_{method}_{suffix}.png'), stretch_canvas_warp)
        cv2.imwrite(
            os.path.join(output_dir_latest, 'png', f'sewing_pattern_{method}_{suffix}.png'), patch_canvas)
        cv2.imwrite(
            os.path.join(output_dir_method, 'png', f'sewing_pattern_{method}_{suffix}.png'), patch_canvas)
    
    output_dir = os.path.join(output_rootdir, 'dfx', method)
    os.makedirs(output_dir, exist_ok=True)
    process_2d_meshes(
        patches=final_meshes, 
        offsets=MESH_OFFSETS_DICT, 
        output_dir=output_dir,
        method=method
    )


if __name__ == '__main__':
    qualitative_evaluation('average_50.0_10i')
