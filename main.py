from smplx import SMPL
import argparse
import trimesh
import numpy as np

from const import (
    INIT_LEFT_BACK_PANT,
    INIT_LEFT_FRONT_PANT,
    INIT_LEFT_SLEEVE,
    INIT_RIGHT_BACK_PANT,
    INIT_RIGHT_FRONT_PANT,
    INIT_RIGHT_SLEEVE,
    INIT_UPPER_BACK,
    INIT_UPPER_FRONT,
    KEYPOINTS,
    PANT_LENGTH,
    SEAM_IDX_DICT,
    SHIRT_LENGTH,
    SLEEVE_LENGTH
)
from garment import Garment
from geometry import (
    bezier_curve,
    find_init_vertex_idx,
    project_points_to_nearest_faces,
    subdivide_mesh
)
from seams import determine_pant_seams, determine_shirt_seams
from utils import (
    export_to_ply,
    update_color_indices
)
from seams import extract_parameterized_seams
from selector import (
    select_original,
    select_subdivided
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gender', '-G', type=str, choices=['male', 'female', 'neutral'], default='female')
    parser.add_argument('--subdivide', dest='subdivide', action='store_true')
    parser.add_argument('--shirt_length', '-S', type=float, default=SHIRT_LENGTH)
    parser.add_argument('--pant_length', '-P', type=float, default=PANT_LENGTH)
    parser.add_argument('--sleeve_length', '-L', type=float, default=SLEEVE_LENGTH)
    args = parser.parse_args()

    smpl_model = SMPL(model_path='/data/hierprob3d/smpl/SMPL_FEMALE.pkl')

    if args.subdivide:
        select_subdivided(args, smpl_model)
    else:
        select_original(smpl_model)
