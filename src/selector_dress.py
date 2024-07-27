import numpy as np
import os
import json
from smplx import SMPL
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
    update_color_indices,
    extract_local_stretches
)


def generate_original_meshes(smpl_dir, original_dir, set_dict):
    smpl_models = {}
    smpl_models['male'] = SMPL(model_path=os.path.join(smpl_dir, f'SMPL_MALE.pkl'), gender='male')
    smpl_models['female'] = SMPL(model_path=os.path.join(smpl_dir, f'SMPL_FEMALE.pkl'), gender='female')

    os.makedirs(original_dir, exist_ok=True)

    for set_element_idx in range(len(set_dict['poses'])):
        pose_fun = getattr(const, set_dict['poses'][set_element_idx])
        shape_fun = getattr(const, set_dict['shapes'][set_element_idx])
        gender = set_dict['genders'][set_element_idx]
        posed_verts = smpl_models[gender](
            body_pose=pose_fun(), 
            betas=shape_fun()
        ).vertices[0].cpu().detach().numpy()
        mesh_name = 'init' if set_element_idx == 0 else f'target-{set_element_idx:02d}'
        trimesh.Trimesh(vertices=posed_verts, faces=smpl_models[gender].faces).export(os.path.join(original_dir, f'{mesh_name}.ply'))


def load_skirtified_meshes(skirtified_dir):
    meshes = []
    for fname in os.listdir(skirtified_dir):
        fpath = os.path.join(skirtified_dir, fname)
        skirtified_mesh = trimesh.load(fpath)
        meshes.append(skirtified_mesh)
    return meshes


def select_skirtified_dress(args, smpl_dir):
    original_dir = f'data/skirtified/original/{args.design}-{args.body_set}'
    skirtified_dir = f'data/skirtified/skirtified/{args.design}-{args.body_set}/'

    seam_idx_dict = SEAM_IDX_DICT['dress']

    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
            set_dict = json.load(json_file)

    if not os.path.exists(skirtified_dir):
        generate_original_meshes(smpl_dir, original_dir, set_dict)
        print('NOTE: Skirtified meshes not yet created.')
        print('NOTE: Generating original SMPL meshes and exiting...')
        os.makedirs(skirtified_dir, exist_ok=True)
        return
    else:
        skirtified_meshes = load_skirtified_meshes(skirtified_dir)

    init_verts = skirtified_meshes[0].vertices
    faces = skirtified_meshes[0].faces

    garment = Garment(init_verts, faces, skirtification_type='dress')

    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)

    # Determine seams.
    seam_idxs_front, y_shirt_threshold = determine_shirt_seams(
        verts=init_verts, 
        shirt_length=design_dict['dims']['upper'], 
        seam_idx_dict=seam_idx_dict['upper_front']
    )
    seam_idxs_back, _ = determine_shirt_seams(
        verts=init_verts, 
        shirt_length=design_dict['dims']['upper'], 
        seam_idx_dict=seam_idx_dict['upper_back']
    )

    # Flood fill.
    front_v_idxs = garment.flood_fill_vertices(
        vertex_positions=init_verts, 
        boundary_vertices=seam_idxs_front, 
        y_threshold=y_shirt_threshold, 
        start_vertex=INIT_UPPER_FRONT_SKIRTIFIED
    )
    back_v_idxs = garment.flood_fill_vertices(
        vertex_positions=init_verts, 
        boundary_vertices=seam_idxs_back, 
        y_threshold=y_shirt_threshold, 
        start_vertex=INIT_UPPER_BACK_SKIRTIFIED
    )

    for offset_type in ['skintight', 'loose']:
        for set_element_idx in range(len(set_dict['poses'])):
            posed_verts = skirtified_meshes[set_element_idx].vertices
            upper_indices = front_v_idxs + back_v_idxs

            upper_garment_verts, upper_garment_faces = garment.extract_garment_mesh(posed_verts, faces, upper_indices, offset=DISPLACEMENTS[offset_type])
            upper_front_verts, upper_front_faces = garment.extract_garment_mesh(posed_verts, faces, front_v_idxs, offset=DISPLACEMENTS[offset_type], segment_name='upper_front')
            upper_back_verts, upper_back_faces = garment.extract_garment_mesh(posed_verts, faces, back_v_idxs, offset=DISPLACEMENTS[offset_type], segment_name='upper_back')

            # Export garment component meshes
            mesh_name = 'init' if set_element_idx == 0 else f'target-{set_element_idx:02d}'
            mesh_set_dir = os.path.join('data/embedded/', f'{args.design}-{args.body_set}/{offset_type}')
            latest_set_dir = os.path.join('data/embedded/latest/', offset_type)

            os.makedirs(os.path.join(mesh_set_dir, 'upper_front/'), exist_ok=True)
            os.makedirs(os.path.join(mesh_set_dir, 'upper_back/'), exist_ok=True)

            os.makedirs(os.path.join(latest_set_dir, 'upper_front/'), exist_ok=True)
            os.makedirs(os.path.join(latest_set_dir, 'upper_back/'), exist_ok=True)

            export(args, upper_front_verts, upper_front_faces, f'{mesh_set_dir}/upper_front/{mesh_name}', args.file_format)
            export(args, upper_back_verts, upper_back_faces, f'{mesh_set_dir}/upper_back/{mesh_name}', args.file_format)

            export(args, upper_front_verts, upper_front_faces, f'{latest_set_dir}/upper_front/{mesh_name}', args.file_format)
            export(args, upper_back_verts, upper_back_faces, f'{latest_set_dir}/upper_back/{mesh_name}', args.file_format)

            # Prepare local stretch arrays
            upper_front_stretch_array_u, upper_front_stretch_array_v = extract_local_stretches(
                verts=upper_front_verts,
                faces=upper_front_faces,
                design_dict=design_dict['stretches'],
                garment_part='upper'
            )
            upper_back_stretch_array_u, upper_back_strech_array_v = extract_local_stretches(
                verts=upper_back_verts,
                faces=upper_back_faces,
                design_dict=design_dict['stretches'],
                garment_part='upper'
            )

            np.savetxt(f'{mesh_set_dir}/upper_front/stretches_u.txt', upper_front_stretch_array_u)
            np.savetxt(f'{mesh_set_dir}/upper_back/stretches_u.txt', upper_back_stretch_array_u)

            np.savetxt(f'{mesh_set_dir}/upper_front/stretches_v.txt', upper_front_stretch_array_v)
            np.savetxt(f'{mesh_set_dir}/upper_back/stretches_v.txt', upper_back_strech_array_v)

            np.savetxt(f'{latest_set_dir}/upper_front/stretches_u.txt', upper_front_stretch_array_u)
            np.savetxt(f'{latest_set_dir}/upper_back/stretches_u.txt', upper_back_stretch_array_u)

            np.savetxt(f'{latest_set_dir}/upper_front/stretches_v.txt', upper_front_stretch_array_v)
            np.savetxt(f'{latest_set_dir}/upper_back/stretches_v.txt', upper_back_strech_array_v)

            # Store for the initial color-coded designs.
            if mesh_name == 'init' and offset_type == 'skintight':
                upper_garment_stretch_array_u, upper_garment_stretch_array_v = extract_local_stretches(
                    verts=upper_garment_verts,
                    faces=upper_garment_faces,
                    design_dict=design_dict['stretches'],
                    garment_part='upper'
                )
                
                upper_garment_mesh = color_code_stretches(
                    verts=upper_garment_verts, 
                    faces=upper_garment_faces,
                    stretch_array=upper_garment_stretch_array_u
                )
                upper_garment_mesh.export(f'{mesh_set_dir}/upper_garment_{mesh_name}.ply')
                upper_garment_mesh.export(f'{latest_set_dir}/upper_garment_{mesh_name}.ply')

            export(args, init_verts, faces, f'{mesh_set_dir}/body-{set_element_idx:02d}', args.file_format)
            export(args, init_verts, faces, f'{latest_set_dir}/body-{set_element_idx:02d}', args.file_format)

            garment.store_seamline_vertex_pairs(subdir=f'{args.design}-{args.body_set}')
