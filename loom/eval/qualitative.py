from typing import List, Dict, Tuple
from copy import deepcopy
import cv2
import os
import numpy as np

from loom.eval.vis import (
    QualitativeMesh,
    PatchProcessor,
    mesh_to_image,
    draw_mesh_on_canvas
)
from loom.eval.stretch_utils import (
    calculate_vertex_colors,
    interpolate_color
)
from loom.eval.const import (
    MESH_OFFSETS_DICT,
    GLOBAL_IMG_SCALE
)


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

def mesh_to_stretch_image(mesh, face_stretches, image_size=(800, 800), 
                         min_stretch=0.85, max_stretch=1.15):
    """
    Create a smoothly interpolated visualization of stretch values.
    
    Args:
        mesh: Trimesh object containing the mesh
        face_stretches: Array of stretch values per face
        image_size: Size of output image
        min_stretch/max_stretch: Range for color mapping
    
    Returns:
        Image array with smooth stretch visualization
    """
    # Project vertices to 2D as in your original code
    points = np.array(mesh.vertices)
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    points[:, :2] = (points[:, :2] - min_bounds[:2]) * GLOBAL_IMG_SCALE
    
    # Calculate vertex colors
    vertex_colors = calculate_vertex_colors(mesh, face_stretches, min_stretch, max_stretch)
    
    # Create output image
    image = np.zeros((*image_size, 4), dtype=np.uint8)
    
    # For each triangle
    for face_idx, face in enumerate(mesh.faces):
        # Get 2D coordinates of triangle vertices
        tri_points = points[face, :2].astype(int)
        
        # Get bounding box of triangle
        min_x = max(tri_points[:, 0].min(), 0)
        max_x = min(tri_points[:, 0].max() + 1, image_size[0])
        min_y = max(tri_points[:, 1].min(), 0)
        max_y = min(tri_points[:, 1].max() + 1, image_size[1])
        
        # For each pixel in bounding box
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                point = np.array([x, y])
                
                # Check if point is inside triangle
                if cv2.pointPolygonTest(tri_points, (x, y), False) >= 0:
                    # Interpolate color
                    color = interpolate_color(
                        point, 
                        tri_points, 
                        vertex_colors[face]
                    )
                    image[y, x] = color
    
    return image[::-1]


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
        output_rootdir: str,
        experiment_name: str
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
    
    #output_dir_latest = os.path.join(output_rootdir, 'latest/')
    output_dir_method = os.path.join(output_rootdir, 'png', experiment_name)
    #os.makedirs(os.path.join(output_dir_latest, 'png'), exist_ok=True)     
    os.makedirs(output_dir_method, exist_ok=True)     
    
    #final_meshes = deepcopy(meshes_dict['pre'])
    final_meshes = deepcopy(meshes_dict['final-seams'])
    
    for suffix in meshes_dict:
        meshes_dict[suffix].sort(key=lambda mesh: mesh.is_front)

        canvas_size = (1200, 1200)
        img_size = (450, 450)
        
        # For stretch visualization
        #stretch_canvas_weft = combine_meshes_on_canvas(
        #    meshes_dict[suffix], canvas_size, img_size, visualization_mode='stretch', direction='weft')
        #stretch_canvas_warp = combine_meshes_on_canvas(
        #    meshes_dict[suffix], canvas_size, img_size, visualization_mode='stretch', direction='warp')

        # For original front/back visualization
        #patch_canvas = combine_meshes_on_canvas(
        #    meshes_dict[suffix], canvas_size, img_size, visualization_mode='sewing_pattern')

        #cv2.imwrite(
        #    os.path.join(output_dir_latest, 'png', f'weft_stretch_pattern_{experiment_name}_{suffix}.png'), stretch_canvas_weft)
        #cv2.imwrite(
        #    os.path.join(output_dir_method, 'png', f'weft_stretch_pattern_{experiment_name}_{suffix}.png'), stretch_canvas_weft)
        #cv2.imwrite(
        #    os.path.join(output_dir_latest, 'png', f'warp_stretch_pattern_{experiment_name}_{suffix}.png'), stretch_canvas_warp)
        #cv2.imwrite(
        #    os.path.join(output_dir_method, 'png', f'warp_stretch_pattern_{experiment_name}_{suffix}.png'), stretch_canvas_warp)
        #cv2.imwrite(
        #    os.path.join(output_dir_latest, 'png', f'sewing_pattern_{experiment_name}_{suffix}.png'), patch_canvas)
        #cv2.imwrite(
        #    os.path.join(output_dir_method, f'sewing_pattern_{suffix}.png'), patch_canvas)
    
    process_2d_meshes(
        patches=final_meshes, 
        offsets=MESH_OFFSETS_DICT, 
        output_rootdir=output_rootdir,
        experiment_name=experiment_name
    )


if __name__ == '__main__':
    qualitative_evaluation('average_50.0_10i')
