import json
import os
from smplx import SMPL

from tailorlang import const
from tailorlang.const import (
    COMPONENT_SIGN_DICT,
    DISPLACEMENTS,
    INIT_IDXS,
    INIT_UPPER_FRONT,
    INIT_UPPER_BACK,
    INIT_FRONT_RIGHT_SLEEVE,
    INIT_FRONT_LEFT_SLEEVE,
    INIT_BACK_LEFT_SLEEVE,
    INIT_BACK_RIGHT_SLEEVE,
    INIT_LEFT_BACK_PANT,
    INIT_RIGHT_BACK_PANT,
    INIT_LEFT_FRONT_PANT,
    INIT_RIGHT_FRONT_PANT,
    PRE_SEAMS_DICT,
    SEGMENT_SETS,
    FIXED_POINTS_DICT,
    PLANE_ORIENT_DICT
)
from tailorlang.garment import Garment
from tailorlang.geometry import modify_mesh_with_plane_cut
from tailorlang.utils import (
    create_directories,
    export,
    export_body_mesh
)


def init_static(args):
    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)
    smpl_model = SMPL(model_path=os.path.join(args.smpl_dir, f'SMPL_{args.gender.upper()}.pkl'), gender=args.gender)
    canonical_verts = smpl_model().vertices[0].cpu().detach().numpy()
    
    return design_dict, set_dict, smpl_model, canonical_verts


def init_garment_for_preselect(female_model):
    with open(f'config/designs/full.json', 'r') as json_file:
        design_dict = json.load(json_file)

    verts = female_model().vertices[0].cpu().detach().numpy()
    faces = female_model.faces
    garment = Garment(verts, faces, use_darts=False)

    return garment, design_dict


def initialize_garment_and_configs(args, female_model):
    with open(f'config/designs/{args.design}.json', 'r') as json_file:
        design_dict = json.load(json_file)
    with open(f'config/body_sets/{args.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)

    verts = female_model().vertices[0].cpu().detach().numpy()
    faces = female_model.faces
    garment = Garment(verts, faces, use_darts=args.use_darts)

    return garment, design_dict, set_dict


def process_garment_set(args, modified_models, garment, set_dict, garment_parts, offset_type):
    for set_element_idx in range(len(set_dict['poses'])):
        pose_fun = getattr(const, set_dict['poses'][set_element_idx])
        shape_fun = getattr(const, set_dict['shapes'][set_element_idx])
        gender = set_dict['genders'][set_element_idx]
        
        posed_verts = modified_models[gender](
            body_pose=pose_fun(), 
            betas=shape_fun()
        ).vertices[0].cpu().detach().numpy()

        mesh_name = 'init' if set_element_idx == 0 else f'target-{set_element_idx:02d}'
        mesh_set_dir = os.path.join('data/embedded/', f'{args.design}-{args.body_set}/{offset_type}')
        latest_set_dir = os.path.join('data/embedded/latest/', offset_type)

        create_directories(mesh_set_dir, latest_set_dir)
        garment_parts = process_garment_parts(garment, garment_parts, posed_verts, offset_type)
        
        for part_name in garment_parts:
            part_verts = garment_parts[part_name]['verts']
            part_faces = garment_parts[part_name]['faces']
            export(args, part_verts, part_faces, f'{mesh_set_dir}/{part_name}/{mesh_name}', args.file_format)
            export(args, part_verts, part_faces, f'{latest_set_dir}/{part_name}/{mesh_name}', args.file_format)
            
        export_body_mesh(args, posed_verts, modified_models[gender].faces, set_element_idx, mesh_set_dir, latest_set_dir, gender)


def flood_fill_all_parts(garment, modified_verts, seams_info):
    garment_parts = {}
    
    # Upper garment (shirt) flood fill
    garment_parts['upper_front'] = garment.flood_fill_vertices(
        vertex_positions=modified_verts, 
        boundary_vertices=seams_info['upper_front'], 
        y_threshold=seams_info['y_upper_threshold'], 
        start_vertex=INIT_UPPER_FRONT
    )
    garment_parts['upper_back'] = garment.flood_fill_vertices(
        vertex_positions=modified_verts, 
        boundary_vertices=seams_info['upper_back'], 
        y_threshold=seams_info['y_upper_threshold'], 
        start_vertex=INIT_UPPER_BACK
    )

    # Sleeve flood fill
    sleeve_init_points = {
        'sleeve_front_right': INIT_FRONT_RIGHT_SLEEVE,
        'sleeve_back_right': INIT_BACK_RIGHT_SLEEVE,
        'sleeve_front_left': INIT_FRONT_LEFT_SLEEVE,
        'sleeve_back_left': INIT_BACK_LEFT_SLEEVE
    }

    for key, init_idx in sleeve_init_points.items():
        garment_parts[key] = garment.flood_fill_sleeve_vertices(
            vertex_positions=modified_verts, 
            boundary_vertices=seams_info[key]['seams'], 
            start_vertex=init_idx, 
            x_threshold=seams_info[key]['threshold'], 
            side=key.split('_')[-1]
        )

    # Lower garment (pant) flood fill
    lower_init_points = {
        'lower_front_right': INIT_RIGHT_FRONT_PANT,
        'lower_front_left': INIT_LEFT_FRONT_PANT,
        'lower_back_right': INIT_RIGHT_BACK_PANT,
        'lower_back_left': INIT_LEFT_BACK_PANT
    }

    for key, init_idx in lower_init_points.items():
        garment_parts[key] = garment.flood_fill_vertices(
            vertex_positions=modified_verts, 
            boundary_vertices=seams_info[key]['seams'],
            y_threshold=seams_info[key]['threshold_low'], 
            start_vertex=init_idx,
            up_pant_threshold=seams_info[key]['threshold_up']
        )

    return garment_parts


def process_garment_parts(garment, garment_parts, posed_verts, offset_type):
    garment_parts = {}
    for part_name in SEGMENT_SETS['default']:
        part_verts, part_faces = garment.extract_garment_mesh(
            posed_verts, 
            garment.mesh.faces, 
            garment_parts[part_name],
            offset=DISPLACEMENTS[offset_type], 
            segment_name=part_name
        )
        garment_parts[part_name] = {
            'verts': part_verts,
            'faces': part_faces
        }
    return garment_parts


def collect_vert_idxs_of_full_garments(patch_idxs_dict):
    patch_idxs_dict['upper'] = \
            patch_idxs_dict['upper_front'] + patch_idxs_dict['upper_back'] + \
            patch_idxs_dict['sleeve_front_right'] + patch_idxs_dict['sleeve_back_right'] + \
            patch_idxs_dict['sleeve_front_left'] + patch_idxs_dict['sleeve_back_left']
    patch_idxs_dict['lower'] = \
            sum([patch_idxs_dict[f'lower_{part}'] for part in ['front_right', 'front_left', 'back_right', 'back_left']], [])
    return patch_idxs_dict
    
    
def apply_length_params(design_dict, verts, smpl_model):
    threshold_dict = {}
    for component_label in ['upper', 'sleeve_left', 'sleeve_right', 'lower']:
        component_length = design_dict['dims'][component_label.split('_')[0]]
        threshold_dict[component_label] = FIXED_POINTS_DICT[component_label] + COMPONENT_SIGN_DICT[component_label] * component_length
        modified_verts = modify_mesh_with_plane_cut(
            vertices=verts,
            faces=smpl_model.faces,
            cutting_point=threshold_dict[component_label],
            plane_orientation=PLANE_ORIENT_DICT[component_label],
            sleeve_side=component_label.split('_')[1] if component_label[:6] == 'sleeve' else None
        )
    return modified_verts, threshold_dict


def flood_fill(self, vertex_positions, boundary_vertices, start_vertex, garment_label, threshold, **kwargs):
    """
    Unified flood fill function that automatically handles different garment types.
    
    Parameters:
    - vertex_positions: Array of vertex positions
    - boundary_vertices: List of boundary vertex indices
    - start_vertex: Starting vertex index for flood fill
    - garment_label: String indicating garment type ('upper', 'lower', or 'sleeve')
    - threshold: Main threshold value (interpreted based on garment_label)
    - **kwargs: Additional arguments:
        - upper_threshold: Optional upper bound for Y threshold (for upper/lower garments)
        - side: 'left' or 'right' for sleeves
    
    Returns:
    - List of selected vertex indices
    """
    def threshold_check(pos, idx):
        if garment_label == 'sleeve':
            side = kwargs.get('side', 'right')
            return pos[idx, 0] > threshold if side == 'right' else pos[idx, 0] < threshold
        else:
            # For upper and lower garments, use Y threshold
            condition = pos[idx, 1] >= threshold
            upper_threshold = kwargs.get('upper_threshold')
            if upper_threshold is not None:
                condition = condition and pos[idx, 1] <= upper_threshold
            return condition

    # Convert boundary vertices to a set for efficient lookup
    boundary_set = set(boundary_vertices)
    
    # Initialize stack, visited set, and selected vertices set
    stack = [start_vertex]
    visited = set()
    selected_vertices = set()
    
    # Main flood fill loop
    while stack:
        vertex_idx = stack.pop()
        
        # Check if vertex should be processed
        if (vertex_idx not in visited and 
            vertex_idx not in boundary_set and 
            threshold_check(vertex_positions, vertex_idx)):
            
            # Mark vertex as visited and selected
            visited.add(vertex_idx)
            selected_vertices.add(vertex_idx)
            
            # Add unvisited neighbors to stack
            for neighbor_idx in self.vertex_adjacency_list[vertex_idx]:
                if neighbor_idx not in visited:
                    stack.append(neighbor_idx)
    
    # Add qualifying boundary vertices
    thresh_boundaries = [idx for idx in boundary_vertices 
                        if threshold_check(vertex_positions, idx)]
    selected_vertices.update(thresh_boundaries)
    
    return list(selected_vertices)


def build_vertex_adjacency_list(F):
    adjacency_list = {}
    for triangle in F:
        for vertex in triangle:
            if vertex not in adjacency_list:
                adjacency_list[vertex] = set()
            # Add the neighboring vertices
            adjacency_list[vertex].update(triangle)
            adjacency_list[vertex].remove(vertex)  # A vertex is not a neighbor to itself
    return adjacency_list


def apply_pre_seams(modified_verts, faces, threshold_dict):
    vertex_adjacency_list = build_vertex_adjacency_list(faces)
    patch_idxs_dict = {}
    for patch_label in INIT_IDXS:
        patch_type = patch_label.split('_')[0]
        patch_vert_idxs = flood_fill(
            vertex_adjacency_list=vertex_adjacency_list,
            vertex_positions=modified_verts,
            boundary_vertices=PRE_SEAMS_DICT[patch_type],
            start_vertex=INIT_IDXS[patch_label],
            patch_type=patch_type,
            threshold=threshold_dict[patch_type],
            side=patch_label.split('_')[2] if patch_type == 'sleeve' else None
        )
        patch_idxs_dict[patch_label] = patch_vert_idxs
    return patch_idxs_dict


def apply_masks(patch_idxs_dict, modified_verts, threshold_dict):
    
    def _upper_mask(verts, threshold_dict):
        return (verts[:, 1] > threshold_dict['upper']) & \
               (verts[:, 0] > threshold_dict['sleeve_right']) & \
               (verts[:, 0] < threshold_dict['sleeve_left'])
               
    def _lower_mask(verts, threshold_dict):
        return verts[:, 1] > threshold_dict['lower']
    
    for patch_label in patch_idxs_dict:
        patch_verts = modified_verts[patch_idxs_dict[patch_label]]
        mask_fun = _upper_mask if patch_label.split('_')[0] == 'upper' else _lower_mask
        mask = mask_fun(patch_verts, threshold_dict)
        patch_verts[patch_label] = patch_verts[patch_label][mask]
        
    return patch_idxs_dict
    

def apply_all_cuts(seams_dict, smpl_model, upper_verts, lower_verts, modified_verts):
    # TODO: Apply other types of cuts (other seamlines, darts, and cuts).
    return seams_dict
