from typing import Any, Dict
from types import SimpleNamespace
import argparse
import os
import json
from pprint import pformat

from tailorlang.mesh_processing import MeshState
from tailorlang.vis.pattern import visualize_pattern


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str,
                        help='an absolute path to this project')
    parser.add_argument('--smpl_dir', type=str,
                        help='directory containing SMPL models')
    parser.add_argument('--setup_config', type=str, default="default",
                        help='name of the configuration file in config/setup/ directory')
    parser.add_argument('--body_set', type=str,
                        help='body set specification')
    parser.add_argument('--design', type=str,
                        help='design specification')
    parser.add_argument('--optim_dress', action='store_true',
                        help='enable dress optimization')
    parser.add_argument('--apply_remesh', action='store_true',
                        help='enable remeshing')
    parser.add_argument('--use_darts', action='store_true',
                        help='enable use of darts')
    parser.add_argument('--equalize_seamline_lengths', action='store_true',
                        help='enable seamline length equalization')
    parser.add_argument('--matching_mode', type=str,
                        help='matching mode specification')
    parser.add_argument('--seamline_strategy', type=str,
                        help='seamline strategy specification')
    parser.add_argument('--num_seam_iters', type=int,
                        help='number of seam iterations')
    parser.add_argument('--max_stretch', type=float,
                        help='maximum stretch allowed')
    parser.add_argument('--stretch_coef', type=float,
                        help='stretch coefficient')
    parser.add_argument('--edges_coef', type=float,
                        help='edges coefficient')
    parser.add_argument('--seams_coef', type=float,
                        help='seams coefficient')
    parser.add_argument('--dart_coef', type=float,
                        help='dart coefficient')
    
    return parser.parse_args()

def load_config_file(config_name: str, project_dir: str) -> dict:
    """Load configuration from JSON file."""
    config_path = os.path.join(project_dir, 'config', 'setup', f'{config_name}.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

def dict_to_namespace(d: dict) -> SimpleNamespace:
    """Convert a dictionary to a SimpleNamespace recursively."""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)

def namespace_to_dict(ns: SimpleNamespace) -> dict:
    """Convert a SimpleNamespace object back to a dictionary recursively."""
    output = {}
    for key, value in vars(ns).items():
        if isinstance(value, SimpleNamespace):
            output[key] = namespace_to_dict(value)
        else:
            output[key] = value
    return output

def print_configuration(config: SimpleNamespace) -> None:
    """
    Print configuration in the specified format.
    
    Args:
        config: Configuration object
        format: Output format ('json', 'plain', or 'dict')
    """
    config_dict = namespace_to_dict(config)
    
    print("\n=== Configuration ===")
    print(pformat(config_dict, indent=2))
    print("==================\n")

def prepare_configuration() -> SimpleNamespace:
    """Prepare final configuration by combining CLI arguments and config file."""
    # Parse command line arguments
    cli_args = parse_arguments()
    
    # Convert CLI args to dictionary, excluding None values
    cli_dict = {k: v for k, v in vars(cli_args).items() if v is not None}
    
    # Load configuration file
    config_file_dict = load_config_file(cli_args.setup_config, cli_args.project_dir)
    
    # Merge configurations, giving precedence to CLI arguments
    final_config = {**config_file_dict, **cli_dict}
    
    # Convert to namespace for dot notation access
    return dict_to_namespace(final_config)


if __name__ == "__main__":
    config = prepare_configuration()
    print_configuration(config)
    #config.use_darts = True

    mesh_state = MeshState(
        body_set=config.body_set,
        use_darts=config.use_darts,
        apply_remesh=config.apply_remesh)
    #mesh_state.update_parameters(design_params=args.design)
    mesh_state.finalize()
    mesh_state.optimize()

    print('#3 Visualize the optimized pattern...')
    visualize_pattern(method=config.seamline_strategy)
    