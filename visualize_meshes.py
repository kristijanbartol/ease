import trimesh
import pyrender
import numpy as np


VIS_DESIGN_PIPELINE = False
VIS_BODY = True


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
        body_mesh = trimesh.load('data/embedded/latest/skintight/body-02.ply')
        meshes = [body_mesh]

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

    for mesh in meshes:
        mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False))
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
