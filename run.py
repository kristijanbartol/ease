import argparse
import os
from shutil import rmtree

from tailorlang.submodules import run_parameterization
from tailorlang.io import (
    export_edge_lengths
)
from tailorlang.mesh_processing import MeshState
from tailorlang.vis import visualize_pattern


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_darts', action='store_true', dest='use_darts',
                        help='whether to use darts in the design and parameterization algorithm')
    parser.add_argument('--file_format', '-F', type=str, choices=['ply', 'obj', 'both'], default='ply',
                        help='')
    parser.add_argument('--design', '-D', type=str, default='default')
    parser.add_argument('--body_set', type=str, default="set2")
    parser.add_argument('--project_dir', type=str, default='/home/kristijan/TailorLang/', 
                        help='an absolute path to this project')
    parser.add_argument('--smpl_dir', type=str, default="/home/kristijan/data/smpl/models/")
    parser.add_argument('--standard_export', action='store_true', dest='standard_export')
    args = parser.parse_args()

    mesh_state = MeshState(body_set=args.body_set)
    #mesh_state.update_parameters(design_params=args.design)
    mesh_state.finalize()
    mesh_state.optimize()

    print('#3 Visualize the optimized pattern...')
    visualize_pattern()
    