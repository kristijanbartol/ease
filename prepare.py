# Steps to prepare the data:
#   1. (Prepare the template manually - modify individual vertex locations)
#   2. (Select the boundary indices manually)
#   3. Load manually-prepared template
#   4. Select the segment and load the corresponding boundary indices
#   5. Load the initial indices for each segment
#   6. Apply Flood Fill algorithm to obtain the vertex indices representing each segment
#   7. Store the list of vertex indices for each segment in the corresponding NumPy array

import argparse
import os
import numpy as np
from smplx import SMPL

from src.garment import Garment
from src.const import (
    INIT_IDXS,
    SEAM_IDX_DICT,
    SEGMENT_SETS,
    SEGMENTS_DIR
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl_dir', type=str, default='data/smpl/')
    parser.add_argument('--template', type=str, choices=['default'], default='default',
                        help='which template to start with (based on possibly modified SMPL vertex locations)')
    parser.add_argument('--segment_set', type=str, choices=['default', 'half'], default='default',
                        help='which segment set to prepare (important for loading the vertex boundary indices)')
    args = parser.parse_args()

    # NOTE: Loading the SMPL mesh is temporary here - will load from a template ('default' by default, OBJ file).
    # NOTE: The gender is not important when extracting the segments.
    smpl_path = os.path.join(args.smpl_dir, 'SMPL_FEMALE.pkl')
    smpl_model = SMPL(model_path=smpl_path, gender='female')
    verts = smpl_model().vertices[0].cpu().detach().numpy()
    faces = smpl_model.faces

    garment = Garment(verts, faces)

    for segment_label in SEGMENT_SETS[args.segment_set]:
        boundary_verts = []
        for boundary_name in SEAM_IDX_DICT[segment_label]:
            boundary_verts += SEAM_IDX_DICT[segment_label][boundary_name]
        segment_idxs = garment.flood_fill_vertices_simplified(
            boundary_vertices=boundary_verts,
            start_vertex=INIT_IDXS[segment_label]
        )
