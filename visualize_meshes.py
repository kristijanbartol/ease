import trimesh
import pyrender
import numpy as np
import os


VIS_DESIGN_PIPELINE = False
VIS_BODY = True
VIS_DESIGNS = False


def set_mesh_color(garment_mesh, color):
    vertex_colors = np.tile(color, (garment_mesh.vertices.shape[0], 1))
    garment_mesh.visual.vertex_colors = vertex_colors


if __name__ == '__main__':
    if VIS_DESIGN_PIPELINE:
        original_mesh = trimesh.load('data/skirtified/original/dress-solo-female/init.ply')
        skirtified_mesh = trimesh.load('data/skirtified/skirtified/dress-solo-female/init.ply')
        boundaries_mesh = trimesh.load('data/skirtified/skirtified/dress-solo-female/boundaries.ply')
        patch_mesh = trimesh.load('data/skirtified/skirtified/dress-solo-female/patch.ply')
        garment_mesh = trimesh.load('data/embedded/dress-solo-female/skintight/upper_front/init.ply')

        meshes = [original_mesh, skirtified_mesh, boundaries_mesh, patch_mesh, garment_mesh]

        original_mesh.vertices[:, 0] -= 2.0
        skirtified_mesh.vertices[:, 0] -= 1.0
        patch_mesh.vertices[:, 0] += 1.0
        garment_mesh.vertices[:, 0] += 2.0

        # Set the colors for the garment mesh exceptionally
        garment_mesh.visual.face_colors = np.full((len(garment_mesh.faces), 3), [1.0, 0.5, 0.0])
    if VIS_BODY:
        body_mesh = trimesh.load('data/embedded/latest/skintight/body-01.ply')
        meshes = [body_mesh]

    if VIS_DESIGNS:
        mesh_dir = 'data/simulated/houdini/latest/'
        garment_mesh_names = [
            'dress-long_exp.ply',
            'dress-long.ply',
            'dress-medium_exp.ply',
            'dress-medium.ply',
            'dress-short_exp.ply',
            'dress-short.ply',
            'exp_pants_medium.ply',
            'exp_pants_short.ply',
            'male-long.ply',
            'male-medium.ply',
            'medium.ply',
            'short.ply',
            'exp_shirt_below_chest.ply'
        ]
        body_mesh_names = [
            'female.ply',
            'female.ply',
            'female.ply',
            'female.ply',
            'female.ply',
            'female.ply',
            'female.ply',
            'female.ply',
            'male.ply',
            'male.ply',
            'female.ply',
            'female.ply',
            'female.ply'
        ]
        mesh_offsets = [
            (-3, 0, 0),
            (-2, 0, 0),
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (-2.5, 0, -1),
            (-1.5, 0, -1),
            (-0.5, 0, -1),
            (0.5, 0, -1),
            (1.5, 0, -1),
            (2.5, 0, -1)
        ]
        garment_mesh_paths = [os.path.join(mesh_dir, x) for x in garment_mesh_names]
        body_mesh_paths = [os.path.join(mesh_dir, x) for x in body_mesh_names]

        meshes = []
        for garment_idx, mesh_path in enumerate(garment_mesh_paths):
            garment_mesh = trimesh.load(mesh_path)
            garment_mesh.vertices[:, 0] += mesh_offsets[garment_idx][0]
            garment_mesh.vertices[:, 2] += mesh_offsets[garment_idx][2]
            garment_mesh.visual.vertex_colors = np.tile([1.0, 0.5, 0.0], (garment_mesh.vertices.shape[0], 1))
            meshes.append(garment_mesh)
            body_mesh = trimesh.load(body_mesh_paths[garment_idx])
            body_mesh.vertices[:, 0] += mesh_offsets[garment_idx][0]
            body_mesh.vertices[:, 2] += mesh_offsets[garment_idx][2]
            meshes.append(body_mesh)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

    for mesh in meshes:
        mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=True))
        scene.add_node(mesh_node)

    camera = pyrender.PerspectiveCamera(yfov=(3.14159 / 3.0))
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.5],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)
    pyrender.Viewer(scene, use_raymond_lighting=True)
