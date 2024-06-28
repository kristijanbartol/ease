import numpy as np
import torch
import os
from smplx import SMPL

from src.const import (
    DISPLACEMENTS,
    INIT_IDXS,
    SEGMENT_SETS,
    SEGMENT_TO_THRESH_DICT,
    SEAM_IDX_DICT    
)
import src.const as const
from src.garment import Garment
from src.geometry import cut_seamlines
from src.utils import (
    export,
    extract_local_stretches
)


def load_template(args, smpl_dir):
    # TODO: Could also load the 
    smpl_path = os.path.join(smpl_dir, 'SMPL_FEMALE.pkl')
    smpl_model = SMPL(model_path=smpl_path, gender='female')
    verts = smpl_model().vertices[0].cpu().detach().numpy()
    faces = smpl_model.faces
    return verts, faces


class MeshExtractor():

    def __init__(
            self, 
            smpl_dir, 
            verts, 
            faces, 
            args, 
            config, 
            threshold_dict
        ):
        self.verts = verts
        self.faces = faces
        self.garment = Garment(verts, faces)
        self.male_smpl_model = SMPL(
            os.path.join(smpl_dir, 'SMPL_MALE.pkl'), 
            gender='male', 
            v_template=torch.from_numpy(verts)
        )
        self.female_smpl_model = SMPL(
            os.path.join(smpl_dir, 'SMPL_FEMALE.pkl'), 
            gender='female', 
            v_template=torch.from_numpy(verts)
        )
        self.args = args
        segment_dirname = f'{args.template}-{args.segment_set}'
        self.config = config
        self.threshold_dict = threshold_dict

    def get_segment_mesh(
            self, 
            set_element_idx, 
            segment_label, 
            offset_type
        ):
        boundary_verts = []
        for boundary_name in SEAM_IDX_DICT[segment_label]:
            boundary_verts += SEAM_IDX_DICT[segment_label][boundary_name]
        '''
        segment_idxs = self.garment.flood_fill_vertices_simplified(
            boundary_vertices=boundary_verts,
            start_vertex=INIT_IDXS[segment_label]
        )
        segment_idxs, segment_faces = self.garment.extract_segment_indices(
            verts=self.verts, 
            faces=self.faces,
            garment_vertex_indices=segment_idxs,
            threshold_dict=self.threshold_dict,
            segment_label=segment_label
        )
        '''
        if 'sleeve' not in segment_label:
            segment_idxs = self.garment.flood_fill_vertices(
                vertex_positions=self.verts,
                boundary_vertices=boundary_verts,
                y_threshold=self.threshold_dict[SEGMENT_TO_THRESH_DICT[segment_label]],
                start_vertex=INIT_IDXS[segment_label]
            )
        else:
            segment_idxs = self.garment.flood_fill_sleeve_vertices(
                vertex_positions=self.verts,
                boundary_vertices=boundary_verts,
                start_vertex=INIT_IDXS[segment_label],
                x_threshold=self.threshold_dict[SEGMENT_TO_THRESH_DICT[segment_label]],
                side=segment_verts.split('_')[-1]
            )

        pose_fun = getattr(const, self.config['body_set']['poses'][set_element_idx])
        betas = getattr(const, self.config['body_set']['shapes'][set_element_idx])()
        gender = self.config['body_set']['genders'][set_element_idx]
        smpl_model = self.male_smpl_model if gender == 'male' else self.female_smpl_model
        segment_verts = smpl_model(
            body_pose=pose_fun(), 
            betas=betas
        ).vertices[0].cpu().detach().numpy()

        segment_verts, segment_faces = self.garment.extract_garment_mesh(
            verts=segment_verts, 
            faces=self.faces,
            garment_vertex_indices=segment_idxs,
            offset=DISPLACEMENTS[offset_type]
        )
        return segment_verts, segment_faces
    
    def save_local_stretches(
            self, 
            mesh_set_dirname, 
            segment_label, 
            segment_verts, 
            segment_faces, 
        ):
        stretch_array_u, stretch_array_v = extract_local_stretches(
            verts=segment_verts,
            faces=segment_faces,
            design_dict=self.config['design']['stretches'],
            garment_part=segment_label.split('_')[0],
            side=segment_label.split('_')[-1]
        )
        segment_dir = os.path.join(mesh_set_dirname, segment_label)
        np.savetxt(os.path.join(segment_dir, 'stretches_u.txt'), stretch_array_u)
        np.savetxt(os.path.join(segment_dir, 'stretches_v.txt'), stretch_array_v)

    def save_mesh_data(
            self, 
            set_element_idx, 
            mesh_set_dirname,
            segment_label, 
            latest_set_dir,
            segment_verts, 
            segment_faces
        ):
        segment_dir = os.path.join(mesh_set_dirname, segment_label)
        export(
            args=self.args, 
            verts=segment_verts, 
            faces=segment_faces, 
            path=os.path.join(segment_dir, f'target-{set_element_idx:02d}'),
            format=self.args.file_format
        )
        export(self.args, self.verts, self.faces, f'{mesh_set_dirname}/body-{set_element_idx:02d}', self.args.file_format)
        export(self.args, self.verts, self.faces, f'{latest_set_dir}/body-{set_element_idx:02d}', self.args.file_format)


# NOTE: For the skirtified meshes, it is best to create a separate function as each step is slightly different.
def generate_garment_components(args, smpl_dir, config):
    # 1. Load template mesh (default, half, skirtified, ...)
    verts, faces = load_template(args, smpl_dir)
    # 2. Cut the edge seamlines (modify geometry)
    modified_verts, threshold_dict = cut_seamlines(
        design_dict=config['design'],
        verts=verts,
        faces=faces
    )
    # 3. Create male and female SMPL models based on modified geometry
    mesh_extractor = MeshExtractor(
        smpl_dir=smpl_dir,
        verts=modified_verts,
        faces=faces,
        args=args,
        config=config,
        threshold_dict=threshold_dict
    )

    for offset_type in ['skintight', 'loose']:
        for segment_label in SEGMENT_SETS[args.segment_set]:
            for body_set_element_idx in range(len(config['body_set']['poses'])):
                # 4. Extract segmented mesh component
                segment_verts, segment_faces = mesh_extractor.get_segment_mesh(
                    set_element_idx=body_set_element_idx,
                    segment_label=segment_label,
                    offset_type=offset_type
                )
                # 5. Save local stretches for the initial mesh
                mesh_set_dirname = os.path.join(f'data/{args.design}-{args.set}/{offset_type}')
                latest_set_dir = os.path.join(f'data/latest/{offset_type}')
                if body_set_element_idx == 0:
                    mesh_extractor.save_local_stretches(
                        mesh_set_dirname=mesh_set_dirname,
                        segment_label=segment_label,
                        segment_verts=segment_verts
                    )
                # 6. Save posed segmented mesh
                mesh_extractor.save_mesh_data(
                    set_element_idx=body_set_element_idx,
                    mesh_set_dirname=mesh_set_dirname,
                    segment_label=segment_label,
                    latest_set_dir=latest_set_dir,
                    segment_verts=segment_verts,
                    segment_faces=segment_faces
                )
