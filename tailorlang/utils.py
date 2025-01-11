from typing import Any, Dict
from types import SimpleNamespace
import argparse
import os
import json
from pprint import pformat
from copy import deepcopy

from tailorlang.const import EXPERIMENT_GROUPS


def construct_experiment_name(config):
    method_name = f'{config.body_set}_{config.design}_{config.seamline_strategy}_{config.num_seam_iters}_{config.stretch_coef}_{config.edges_coef}_{config.seams_coef}'
    method_name += '_' + 'T' if config.apply_remesh else 'F'
    method_name += 'T' if config.use_darts else 'F'
    method_name += 'T' if config.equalize_seamline_lengths else 'F'
    
    return method_name


def construct_configs(init_config):
    group_name = init_config.group_label
    if group_name not in EXPERIMENT_GROUPS:
        return [init_config]
    else:
        grid_parameters = EXPERIMENT_GROUPS[group_name]
        
    configs = []
    
    for body_set in grid_parameters['body_sets']:
        for design_set in grid_parameters['designs']:    
        #for matching_mode in grid_parameters['matching_modes']:
            for seamline_strategy in grid_parameters['seamline_strategies']:
                for num_seam_iters in grid_parameters['num_seam_iterss']:
                    for stretch_coef in grid_parameters['stretch_coefs']:
                        for edges_coef in grid_parameters['edge_coefs']:
                            for seams_coef in grid_parameters['seam_coefs']:
                                for _apply_remesh in grid_parameters['apply_remesh']:
                                    for _use_darts in grid_parameters['use_darts']:
                                        for _equalize_seamline_lengths in grid_parameters['equalize_seamline_lengths']:
                                            _config = deepcopy(init_config)
                                            
                                            _config.body_set = body_set
                                            _config.design_set = design_set
                                            _config.seamline_strategy = seamline_strategy
                                            _config.num_seam_iters = num_seam_iters
                                            _config.stretch_coef = stretch_coef
                                            _config.edges_coef = edges_coef
                                            _config.seams_coef = seams_coef
                                            _config.apply_remesh = _apply_remesh
                                            _config.use_darts = _use_darts
                                            _config.equalize_seamline_lengths = _equalize_seamline_lengths
                                            
                                            configs.append(_config)
    return configs


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

def get_experiment_names_for_grid(init_config):
    grid_configs = construct_configs(init_config=init_config)
    experiment_names = []
    for config in grid_configs:
        experiment_names.append(construct_experiment_name(config))
    return experiment_names
