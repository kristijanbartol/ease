import open3d as o3d
import numpy as np
import cv2
import os
import argparse


class Mesh:
    def __init__(self, ply_path, offset, is_front):
        self.mesh = o3d.io.read_triangle_mesh(ply_path)
        self.offset = offset
        self.is_front = is_front


def mesh_to_image(mesh, image_size=(800, 800)):
    # Project the 3D mesh to 2D
    points = np.asarray(mesh.vertices)
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    
    # Normalize the points to the range [0, image_size]
    scale = min(image_size) / (max_bounds[:2] - min_bounds[:2]).max()
    points[:, :2] = (points[:, :2] - min_bounds[:2]) * scale

    # Create an empty image with alpha channel (transparent background)
    image = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)

    triangles = np.asarray(mesh.triangles)
    for tri in triangles:
        pts = points[tri][:, :2].astype(int)
        cv2.fillConvexPoly(image, pts, (0, 0, 0, 255))  # Black color for the mesh
    
    return image[::-1]


def draw_mesh_on_canvas(canvas, mesh_image, offset, color):
    center = int(canvas.shape[0] / 2), int(canvas.shape[1] / 2)
    half_sizes = int(mesh_image.shape[0] / 2), int(mesh_image.shape[1] / 2)
    x_offset, y_offset = offset
    y1, y2 = center[1] + y_offset - half_sizes[1], center[1] + y_offset + half_sizes[1]
    x1, x2 = center[0] + x_offset - half_sizes[0], center[0] + x_offset + half_sizes[0]

    # Extract color channels if the image has an alpha channel
    if mesh_image.shape[2] == 4:
        img_rgb = mesh_image[..., :3]
        img_alpha = mesh_image[..., 3]
    else:
        img_rgb = mesh_image
        img_alpha = None

    # Apply the color to the mesh while preserving transparency
    colored_img = cv2.addWeighted(img_rgb, 0.5, np.full_like(img_rgb, color), 0.5, 0)
    
    if img_alpha is not None:
        mask = img_alpha / 255.0
        for c in range(0, 3):
            canvas[y1:y2, x1:x2, c] = canvas[y1:y2, x1:x2, c] * (1 - mask) + colored_img[..., c] * mask
        # Update the alpha channel of the canvas
        canvas[y1:y2, x1:x2, 3] = np.maximum(canvas[y1:y2, x1:x2, 3], mesh_image[..., 3])
    else:
        canvas[y1:y2, x1:x2, :3] = colored_img


def combine_meshes(meshes, canvas_size, img_size, front_color=(0, 255, 255), back_color=(0, 165, 255)):
    # Create a blank canvas with alpha channel (transparent background)
    canvas = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)

    for mesh in meshes:
        mesh_image = mesh_to_image(mesh.mesh, image_size=img_size)
        color = front_color if mesh.is_front else back_color
        draw_mesh_on_canvas(canvas, mesh_image, mesh.offset, color)

    return canvas


def save_combined_image(output_path, combined_image):
    cv2.imwrite(output_path, combined_image)



PATTERN_DICT = {
    'front_shirt': [0, 0],
    'back_shirt': [0, -420],
    'front_right_sleeve': [-300, -150],
    'back_right_sleeve': [-260, -200],
    'front_left_sleeve': [320, -180],
    'back_left_sleeve': [300, -220],
    'front_right_pant': [-200, 250],
    'back_right_pant': [-170, 200],
    'front_left_pant': [300, 250],
    'back_left_pant': [350, 200]
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_format', '-F', type=str, choices=['ply', 'obj', 'both'], default='ply')
    parser.add_argument('--design', '-D', type=str, default='default')
    parser.add_argument('--body_set', type=str, default="set2")
    parser.add_argument('--os', type=str, default="linux")
    args = parser.parse_args()

    data_dir = f'data/embedded/latest/skintight/'
    meshes = []

    for _, dirs, _ in os.walk(data_dir):
        for subdir in dirs:
            pattern_fpath = os.path.join(data_dir, subdir, 'optim.ply')
            is_front = True if subdir.split('_')[0] == 'front' else False
            meshes.append(
                Mesh(pattern_fpath, PATTERN_DICT[subdir], is_front)
            )
    meshes.sort(key=lambda mesh: mesh.is_front)

    canvas_size = (1200, 1200)
    img_size = (300, 300)
    combined_image = combine_meshes(meshes, canvas_size, img_size)

    output_path = os.path.join(data_dir, 'sewing_pattern2.png')
    save_combined_image(output_path, combined_image)
