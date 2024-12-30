import os
import json
import numpy as np

from tailorlang.const import PATCH_LIST
from tailorlang.eval.stretch_utils import (
    load_stretch_data,
    calculate_stretch_statistics,
    print_stretch_statistics
)
from tailorlang.eval.seamline_utils import get_seamline_distances
        
        
def analyze_all_patches(method):
    """
    Analyze stretch statistics for individual patches and combined data.
    
    Parameters:
        method: Method identifier
    
    Returns:
        Dictionary containing statistics for each patch and combined statistics
    """
    patch_statistics = {}
    all_optim_u, all_optim_v = [], []
    all_target_u, all_target_v = [], []
    
    # Collect statistics for individual patches
    for patch_label in PATCH_LIST:
        # Load stretch data for current patch
        optim_u, optim_v, target_u, target_v = load_stretch_data(method, patch_label)
        
        # Calculate statistics for current patch
        patch_statistics[patch_label] = calculate_stretch_statistics(
            optim_u=optim_u,
            optim_v=optim_v,
            target_u=target_u,
            target_v=target_v
        )
        print(f"\nStatistics for patch: {patch_label}")
        print_stretch_statistics(patch_statistics[patch_label])
        
        # Collect data for combined analysis
        all_optim_u.append(optim_u)
        all_optim_v.append(optim_v)
        all_target_u.append(target_u)
        all_target_v.append(target_v)
    
    # Combine all arrays using numpy.concatenate
    combined_optim_u = np.concatenate(all_optim_u)
    combined_optim_v = np.concatenate(all_optim_v)
    combined_target_u = np.concatenate(all_target_u)
    combined_target_v = np.concatenate(all_target_v)
    
    # Calculate statistics for combined data
    patch_statistics['all'] = calculate_stretch_statistics(
        optim_u=combined_optim_u,
        optim_v=combined_optim_v,
        target_u=combined_target_u,
        target_v=combined_target_v
    )
    
    print("\nCombined statistics across all patches:")
    print_stretch_statistics(patch_statistics['all'])
    
    return patch_statistics


def save_patch_statistics(method, patch_statistics, base_dir="results/quantitative/stretch/optim"):
    """
    Save patch statistics to JSON files in the specified directory structure.
    
    Parameters:
        method: Method identifier (string)
        patch_statistics: Dictionary containing statistics for each patch and combined data
        base_dir: Root directory for storing results
    """
    # Create method subdirectory
    method_dir = os.path.join(base_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    
    # Save individual patch statistics
    for patch_label, stats in patch_statistics.items():
        if patch_label == 'all':
            # Save combined statistics directly in method directory
            output_path = os.path.join(method_dir, 'all.json')
        else:
            # Create patch subdirectory and save patch-specific statistics
            patch_dir = os.path.join(method_dir, patch_label)
            os.makedirs(patch_dir, exist_ok=True)
            output_path = os.path.join(patch_dir, 'statistics.json')
        
        # Save statistics to JSON file
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=4)


if __name__ == '__main__':
    method = 'latest'
    
    patch_statistics = analyze_all_patches(method=method)
    save_patch_statistics(
        method=method,
        patch_statistics=patch_statistics
    )
    
    seamline_dists_dict = get_seamline_distances()
