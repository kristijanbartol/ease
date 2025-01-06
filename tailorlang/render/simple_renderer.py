import trimesh
import pyrender
import numpy as np
import os


VIS_DESIGN_PIPELINE = False
VIS_BODY = True
VIS_DESIGNS = False
VIS_SIM = False


def set_mesh_color(garment_mesh, color):
    vertex_colors = np.tile(color, (garment_mesh.vertices.shape[0], 1))
    garment_mesh.visual.vertex_colors = vertex_colors


def render_simple():
    body_mesh = trimesh.load('data/body/ref.ply')
    upper_mesh = trimesh.load('results/simulation/mesh/ref_upper.ply')
    lower_mesh = trimesh.load('results/simulation/mesh/ref_lower.ply')

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

    for mesh in [body_mesh, upper_mesh, lower_mesh]:
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
