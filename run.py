from typing import Any, Dict
from types import SimpleNamespace
import argparse
import os
import json
from pprint import pformat

from tailorlang.mesh_processing import MeshState
from eval.qualitative import visualize_pattern


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
    parser.add_argument('--run_grid', action='store_true',
                        help='runs a grid of configuration settings (for detailed evaluation and analyses)')
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
    
    # Parse known args to handle missing optional arguments
    args, unknown = parser.parse_known_args()
    
    # Get the list of arguments that were explicitly set on the command line
    specified_args = {action.dest for action in parser._actions 
                     if action.dest in vars(args) and 
                     vars(args)[action.dest] is not None and
                     vars(args)[action.dest] != action.default}
    
    return args, specified_args

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
    # Parse command line arguments and get explicitly specified arguments
    cli_args, specified_args = parse_arguments()
    
    # Load configuration file
    config_file_dict = load_config_file(cli_args.setup_config, cli_args.project_dir)
    
    # Only include CLI arguments that were explicitly set
    cli_dict = {k: v for k, v in vars(cli_args).items() 
                if k in specified_args}
    
    # Merge configurations, giving precedence to CLI arguments
    final_config = {**config_file_dict, **cli_dict}
    
    # Convert to namespace for dot notation access
    return dict_to_namespace(final_config)


def run_grid(config):
    exhaustive = False
    matching_modes = ['strict']
    seamline_strategies = ['average', 'line', 'bezier']
    num_seam_iterss = [1]    # TODO: Support setting the number of "inner" iterations
    stretch_coefs = [0.0, 2.0, 10.0]
    edge_coefs = [0.0, 1.0, 10.0]
    seam_coefs = [2.0, 35.0, 50.0, 100.0]
    
    apply_remesh = [False]
    use_darts = [False]
    equalize_seamline_lengths = [False]
    
    if exhaustive:
        matching_modes = ['strict', 'general']
        seamline_strategies = ['average', 'line', 'bezier']
        stretch_coefs = [0.0, 2.0, 10.0, 20.0, 100.0]
        edge_coefs = [0.0, 1.0, 10.0, 50.0]
        seam_coefs = [0.0, 2.0, 10.0, 20.0, 35.0, 50.0, 100.0]
        apply_remesh = [False, True]
        use_darts = [False, True]
        equalize_seamline_lengths = [False, True]
        
    for matching_mode in matching_modes:
        for seamline_strategy in seamline_strategies:
            for num_seam_iters in num_seam_iterss:
                for stretch_coef in stretch_coefs:
                    for edge_coef in edge_coefs:
                        for seam_coef in seam_coefs:
                            for _apply_remesh in apply_remesh:
                                for _use_darts in use_darts:
                                    for _equalize_seamline_lengths in equalize_seamline_lengths:
                                        config.matching_mode = matching_mode
                                        config.seamline_strategy = seamline_strategy
                                        config.num_seam_iters = num_seam_iters
                                        config.stretch_coef = stretch_coef
                                        config.edge_coef = edge_coef
                                        config.seam_coef = seam_coef
                                        config.apply_remesh = _apply_remesh
                                        config.use_darts = _use_darts
                                        config.equalize_seamline_lengths = _equalize_seamline_lengths
                                        
                                        mesh_state = MeshState(config=config)
                                        mesh_state.finalize()
                                        mesh_state.optimize()
                                        
                                        method = f'{config.matching_mode}_{config.seamline_strategy}_{config.num_seam_iters}_{config.stretch_coef}_{config.edge_coef}_{config.seam_coef}'
                                        
                                        visualize_pattern(method=method)


if __name__ == "__main__":
    config = prepare_configuration()
    print_configuration(config)
    
    if config.run_grid:
        run_grid(config=config)
    else:
        mesh_state = MeshState(config=config)
        #mesh_state.update_parameters(design_params=config.design)
        mesh_state.finalize()
        mesh_state.optimize()

        visualize_pattern(method=config.seamline_strategy)
    