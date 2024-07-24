import open3d as o3d
import numpy as np
import cv2
import os

GLOBAL_SCALE = 400


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
    
    points[:, :2] = (points[:, :2] - min_bounds[:2]) * GLOBAL_SCALE

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
    'upper_front': [0, 0],
    'upper_back': [0, -300],
    'sleeve_front_right': [-300, -150],
    'sleeve_back_right': [-260, -200],
    'sleeve_front_left': [320, -180],
    'sleeve_back_left': [300, -220],
    'lower_front_right': [-200, 250],
    'lower_back_right': [-170, 200],
    'lower_front_left': [300, 250],
    'lower_back_left': [350, 200]
}


if __name__ == "__main__":
    data_dir = f'data/embedded/latest/skintight/'

    meshes_dict = {}
    for _, dirs, _ in os.walk(data_dir):
        for subdir in dirs:
            for fname in os.listdir(os.path.join(data_dir, subdir)):
                if 'optim' in fname:
                    suffix = fname.split('.')[0][5:]
                    pattern_fpath = os.path.join(data_dir, subdir, fname)
                    is_front = True if subdir.split('_')[1] == 'front' else False
                    if suffix not in meshes_dict:
                        meshes_dict[suffix] = [Mesh(pattern_fpath, PATTERN_DICT[subdir], is_front)]
                    else:
                        meshes_dict[suffix].append(Mesh(pattern_fpath, PATTERN_DICT[subdir], is_front))
    for suffix in meshes_dict:
        meshes_dict[suffix].sort(key=lambda mesh: mesh.is_front)

        canvas_size = (1200, 1200)
        img_size = (300, 300)
        combined_image = combine_meshes(meshes_dict[suffix], canvas_size, img_size)

        output_path = os.path.join(data_dir, f'sewing_pattern{suffix}.png')
        save_combined_image(output_path, combined_image)
