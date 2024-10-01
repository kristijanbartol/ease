import numpy as np
import os
import json
import trimesh

from src.const import (
    DISPLACEMENTS,
    INIT_UPPER_BACK_SKIRTIFIED,
    INIT_UPPER_FRONT_SKIRTIFIED,
    SEAM_IDX_DICT
)
import src.const as const
from src.garment import Garment
from src.seams import determine_shirt_seams
from src.utils import (
    export,
    color_code_stretches,
    extract_local_stretches
)
from src.body_processing import initialize_smpl_models


def generate_original_meshes(smpl_dir, original_dir, set_dict):
    smpl_models = initialize_smpl_models(smpl_dir)
    os.makedirs(original_dir, exist_ok=True)
    original_meshes = []

    for set_element_idx in range(len(set_dict['poses'])):
        pose_fun = getattr(const, set_dict['poses'][set_element_idx])
        shape_fun = getattr(const, set_dict['shapes'][set_element_idx])
        gender = set_dict['genders'][set_element_idx]
        posed_verts = smpl_models[gender](
            body_pose=pose_fun(), 
            betas=shape_fun()
        ).vertices[0].cpu().detach().numpy()
        mesh_name = 'init' if set_element_idx == 0 else f'target-{set_element_idx:02d}'
        mesh = trimesh.Trimesh(vertices=posed_verts, faces=smpl_models[gender].faces)
        mesh.export(os.path.join(original_dir, f'{mesh_name}.ply'))
        original_meshes.append(mesh)

    return original_meshes, smpl_models


def load_skirtified_meshes(skirtified_dir):
    meshes = []
    for fname in os.listdir(skirtified_dir):
        if 'init' in fname or 'target' in fname:    # in case there are some intermediate results stored as well
            fpath = os.path.join(skirtified_dir, fname)
            skirtified_mesh = trimesh.load(fpath)
            meshes.append(skirtified_mesh)
    return meshes


def store_colored_faces(verts, faces, seam_idxs, save_path):
    faces_to_color = []
    for face_idx, face in enumerate(faces):
        if any(vertex in seam_idxs for vertex in face):
            faces_to_color.append(face_idx)

    face_colors = np.full((len(faces), 3), [0.33, 0.33, 0.33])

    orange_color = [1.0, 0.5, 0.0]
    for face_idx in faces_to_color:
        face_colors[face_idx] = orange_color

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.visual.face_colors = face_colors
    mesh.export(save_path)


def process_and_export_skirtified_garment_parts(args, garment, design_dict, garment_parts, posed_verts, offset_type, set_element_idx, mesh_name):
    mesh_set_dir = os.path.join('data/embedded/', f'{args.design}-{args.body_set}/{offset_type}')
    latest_set_dir = os.path.join('data/embedded/latest/', offset_type)

    for part in ['upper_front', 'upper_back']:
        part_verts, part_faces = garment.extract_garment_mesh(
            posed_verts, 
            garment.mesh.faces, 
            garment_parts[part], 
            offset=DISPLACEMENTS[offset_type], 
            segment_name=part
        )

        os.makedirs(os.path.join(mesh_set_dir, part), exist_ok=True)
        os.makedirs(os.path.join(latest_set_dir, part), exist_ok=True)

        export(args, part_verts, part_faces, f'{mesh_set_dir}/{part}/{mesh_name}', args.file_format)
        export(args, part_verts, part_faces, f'{latest_set_dir}/{part}/{mesh_name}', args.file_format)

        export_stretch_arrays(design_dict, part_verts, part_faces, part, mesh_set_dir, latest_set_dir)


def export_original_body_mesh(args, verts, faces, set_element_idx, offset_type):
    mesh_set_dir = os.path.join('data/embedded/', f'{args.design}-{args.body_set}/{offset_type}')
    latest_set_dir = os.path.join('data/embedded/latest/', offset_type)

    export(args, verts, faces, f'{mesh_set_dir}/body-{set_element_idx:02d}', args.file_format)
    export(args, verts, faces, f'{latest_set_dir}/body-{set_element_idx:02d}', args.file_format)


def export_skirtified_color_coded_designs(args, garment, design_dict, garment_parts, posed_verts, offset_type):
    mesh_set_dir = os.path.join('data/embedded/', f'{args.design}-{args.body_set}/{offset_type}')
    latest_set_dir = os.path.join('data/embedded/latest/', offset_type)

    upper_indices = garment_parts['upper_front'] + garment_parts['upper_back']
    upper_garment_verts, upper_garment_faces = garment.extract_garment_mesh(posed_verts, garment.mesh.faces, upper_indices, offset=DISPLACEMENTS[offset_type])

    upper_garment_stretch_array_u, _ = extract_local_stretches(upper_garment_verts, upper_garment_faces, design_dict['stretches'], 'upper')
    
    upper_garment_mesh = color_code_stretches(upper_garment_verts, upper_garment_faces, upper_garment_stretch_array_u)
    upper_garment_mesh.export(f'{mesh_set_dir}/upper_garment_init.ply')
    upper_garment_mesh.export(f'{latest_set_dir}/upper_garment_init.ply')


def export_stretch_arrays(design_dict, verts, faces, part_name, mesh_set_dir, latest_set_dir):
    stretch_array_u, stretch_array_v = extract_local_stretches(
        verts=verts,
        faces=faces,
        design_dict=design_dict['stretches'],
        garment_part='upper'
    )

    np.savetxt(f'{mesh_set_dir}/{part_name}/stretches_u.txt', stretch_array_u)
    np.savetxt(f'{mesh_set_dir}/{part_name}/stretches_v.txt', stretch_array_v)
    np.savetxt(f'{latest_set_dir}/{part_name}/stretches_u.txt', stretch_array_u)
    np.savetxt(f'{latest_set_dir}/{part_name}/stretches_v.txt', stretch_array_v)


def setup_directories(args):
    original_dir = f'data/skirtified/original/{args.design}-{args.body_set}'
    skirtified_dir = f'data/skirtified/skirtified/{args.design}-{args.body_set}/'
    os.makedirs(skirtified_dir, exist_ok=True)
    return original_dir, skirtified_dir


def load_set_dict(args):
    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
        return json.load(json_file)


def initialize_garment_and_configs(args, init_mesh):
    garment = Garment(init_mesh.vertices, init_mesh.faces, skirtification_type='dress')
    
    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
    
    return garment, design_dict


def determine_dress_seams(garment, design_dict):
    seam_idx_dict = SEAM_IDX_DICT['dress']
    seams_info = {}

    seams_info['upper_front'], y_shirt_threshold = determine_shirt_seams(
        verts=garment.mesh.vertices, 
        shirt_length=design_dict['dims']['upper'], 
        seam_idx_dict=seam_idx_dict['upper_front']
    )
    seams_info['upper_back'], _ = determine_shirt_seams(
        verts=garment.mesh.vertices, 
        shirt_length=design_dict['dims']['upper'], 
        seam_idx_dict=seam_idx_dict['upper_back']
    )
    seams_info['y_shirt_threshold'] = y_shirt_threshold

    return seams_info

def flood_fill_dress_parts(garment, seams_info):
    garment_parts = {}
    
    garment_parts['upper_front'] = garment.flood_fill_vertices(
        vertex_positions=garment.mesh.vertices, 
        boundary_vertices=seams_info['upper_front'], 
        y_threshold=seams_info['y_shirt_threshold'], 
        start_vertex=INIT_UPPER_FRONT_SKIRTIFIED
    )
    garment_parts['upper_back'] = garment.flood_fill_vertices(
        vertex_positions=garment.mesh.vertices, 
        boundary_vertices=seams_info['upper_back'], 
        y_threshold=seams_info['y_shirt_threshold'], 
        start_vertex=INIT_UPPER_BACK_SKIRTIFIED
    )
    return garment_parts


def process_skirtified_garment_set(args, skirtified_meshes, original_meshes, garment, design_dict, set_dict, garment_parts, offset_type, smpl_models):
    for set_element_idx in range(len(set_dict['poses'])):
        skirtified_verts = skirtified_meshes[set_element_idx].vertices
        original_verts = original_meshes[set_element_idx].vertices
        mesh_name = 'init' if set_element_idx == 0 else f'target-{set_element_idx:02d}'
        
        process_and_export_skirtified_garment_parts(args, garment, design_dict, garment_parts, skirtified_verts, offset_type, set_element_idx, mesh_name)
        
        gender = set_dict['genders'][set_element_idx]
        export_original_body_mesh(args, original_verts, smpl_models[gender].faces, set_element_idx, offset_type)

        if mesh_name == 'init' and offset_type == 'skintight':
            export_skirtified_color_coded_designs(args, garment, design_dict, garment_parts, skirtified_verts, offset_type)


def select_skirtified_dress(args, smpl_dir):
    original_dir, skirtified_dir = setup_directories(args)
    set_dict = load_set_dict(args)

    original_meshes, smpl_models = generate_original_meshes(smpl_dir, original_dir, set_dict)

    if not os.path.exists(skirtified_dir):
        print('NOTE: Skirtified meshes not yet created.')
        print('NOTE: Original SMPL meshes generated. Please create skirtified meshes and run again.')
        return

    skirtified_meshes = load_skirtified_meshes(skirtified_dir)
    garment, design_dict = initialize_garment_and_configs(args, skirtified_meshes[0])
    
    seams_info = determine_dress_seams(garment, design_dict)
    garment_parts = flood_fill_dress_parts(garment, seams_info)
    
    store_colored_faces(garment.mesh.vertices, garment.mesh.faces, seams_info['upper_front'], os.path.join(skirtified_dir, 'boundaries.ply'))
    store_colored_faces(garment.mesh.vertices, garment.mesh.faces, garment_parts['upper_front'], os.path.join(skirtified_dir, 'patch.ply'))

    for offset_type in ['skintight', 'loose']:
        process_skirtified_garment_set(args, skirtified_meshes, original_meshes, garment, design_dict, set_dict, garment_parts, offset_type, smpl_models)

    garment.store_seamline_vertex_pairs(subdir=f'{args.design}-{args.body_set}')
