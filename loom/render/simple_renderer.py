import trimesh
import pyrender
import numpy as np
import os

from tailorlang.const import PATCH_LIST


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
    

PATCH_COLORS = {
    'upper_front': [1.0, 0.5, 0.0],
    'upper_back': [0.5, 0.5, 0.0],
    'sleeve_front_left': [0.5, 0.0, 0.5],
    'sleeve_front_right': [0.2, 0.7, 0.2],
    'sleeve_back_left': [0.8, 0.3, 0.8],
    'sleeve_back_right': [1.0, 0.3, 0.7],
    'lower_front_left': [0.3, 0.7, 1.0],
    'lower_front_right': [0.8, 0.5, 0.2],
    'lower_back_left': [0.6, 0.2, 0.7],
    'lower_back_right': [0.1, 0.6, 0.3]
}    


def render_patches():   # for embedded garment design Figure
    body_mesh = trimesh.load('data/body/target-00.ply')
    
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])
    
    faces_to_remove_dict = {
        'upper_front': [1001, 932, 371, 302],
        'upper_back': [440, 1063]
    }
    
    patch_meshes = []
    for patch_label in PATCH_LIST:
        patch_mesh = trimesh.load(f'data/embedded/{patch_label}/target-00.ply')
        if patch_label in faces_to_remove_dict:
            new_faces = patch_mesh.faces[~np.isin(np.arange(len(patch_mesh.faces)), faces_to_remove_dict[patch_label])]
            patch_mesh.faces = new_faces
            #patch_mesh.update_faces()
            patch_mesh.remove_unreferenced_vertices()
            
        patch_meshes.append(patch_mesh)
        patch_meshes[-1].visual.vertex_colors = np.tile(PATCH_COLORS[patch_label], (patch_meshes[-1].vertices.shape[0], 1))
        
    for mesh_idx, mesh in enumerate(patch_meshes):
        mesh_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mesh, smooth=True))
        scene.add_node(mesh_node)
        mesh.export(f'/Users/kristijanbartol/TailorLang/{PATCH_LIST[mesh_idx]}.ply')

    camera = pyrender.PerspectiveCamera(yfov=(3.14159 / 3.0))
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.3],
        [0.0, 0.0, 1.0, 5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)
    pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == '__main__':
    render_patches()
