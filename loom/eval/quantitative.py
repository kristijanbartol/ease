import os
import json
import numpy as np

from loom.const import PATCH_LIST
from loom.utils import (
    construct_configs,
    construct_experiment_name,
    prepare_configuration,
    print_configuration
)
from loom.eval.stretch_utils import get_stretch_statistics
from loom.eval.seamline_utils import get_seamline_statistics
from loom.eval.table_utils import StatisticsTable


def save_statistics(experiment_name, stats_dict, base_dir):
    """
    Save patch statistics to JSON files in the specified directory structure.
    
    Parameters:
        experiment_name: Method identifier (string)
        stats_dict: Dictionary containing statistics for each patch and combined data
        base_dir: Root directory for storing results
    """
    # Create method subdirectory
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save individual patch statistics
    for part_label, stats in stats_dict.items():
        if part_label == 'all':
            # Save combined statistics directly in method directory
            output_path = os.path.join(exp_dir, 'all.json')
        else:
            # Create patch subdirectory and save patch-specific statistics
            patch_dir = os.path.join(exp_dir, part_label)
            os.makedirs(patch_dir, exist_ok=True)
            output_path = os.path.join(patch_dir, 'statistics.json')
        
        # Save statistics to JSON file
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=4)
            

def load_statistics(experiment_name):
    json_path = os.path.join('results/quantitative/stretch/optim/', experiment_name, 'all.json')
    with open(json_path) as stats_file:
        stats_dict = json.read(stats_file)
    return stats_dict
            

def quantitative_evaluation(experiment_name):
    stretch_statistics = get_stretch_statistics()
    save_statistics(
        experiment_name=experiment_name,
        stats_dict=stretch_statistics,
        base_dir='results/quantitative/stretch/optim/'
    )
    seam_statistics = get_seamline_statistics()
    save_statistics(
        experiment_name=experiment_name,
        seam_statistics=seam_statistics,
        base_dir='results/quantitative/seamline_alignment/'
    )

            
def quantatitative_evaluate_grid(init_config):
    grid_configs = construct_configs(init_config=init_config)
    
    stats_list = []
    for config in grid_configs:
        experiment_name = construct_experiment_name(config)
        stats_list.append(load_statistics(experiment_name))
        
    table = StatisticsTable(
        stats_list=stats_list,  # List of statistics dictionaries
        stat_group='rel_diff',                  # Statistics group to extract
        metrics=['mean', 'median', 'max'],      # Metrics to include
        directions=['weft_direction'],  # Directions to include
        experiment_names=None,          # Optional names
        precision=4                             # Decimal places
    )
    table.to_markdown(os.path.join(f'results/quantitative/stretch/optim/{init_config.group_label}.md'))
    table.to_html(os.path.join(f'results/quantitative/stretch/optim/{init_config.group_label}.html'))
    
    # TODO: Also process seamline statistics.


if __name__ == '__main__':
    config = prepare_configuration()
    print_configuration(config)
    quantatitative_evaluate_grid(init_config=config)
