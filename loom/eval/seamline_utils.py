import os
import trimesh
import numpy as np

from tailorlang.const import (
    ID_TO_PATCH,
    PATCH_LIST,
    SEAM_TO_PATCH_PAIRS
)
from tailorlang.eval.box_plot_utils import compute_box_plot_stats


def read_seamline(seam_fpath):
    """
    Read seamline indices from a text file.
    Returns two arrays of indices for the first and second patches.
    """
    with open(seam_fpath, 'r') as f:
        # Skip the first two lines containing IDs
        next(f)
        next(f)
        
        # Read the indices
        first_indices = []
        second_indices = []
        for line in f:
            idx1, idx2 = map(int, line.strip().split())
            first_indices.append(idx1)
            second_indices.append(idx2)
    
    return np.array(first_indices), np.array(second_indices)


def procrustes_alignment(points1, points2):
    """
    Calculate Procrustes alignment between two sets of points.
    
    Args:
        points1: Target points (N x 3 numpy array)
        points2: Source points to be aligned (N x 3 numpy array)
    
    Returns:
        R: Rotation matrix (3 x 3)
        t: Translation vector (3,)
    """
    # Transpose points for easier centroid computation
    points1t = points1.T
    points2t = points2.T
    
    # Compute centroids
    pb = np.mean(points1t, axis=1)
    qb = np.mean(points2t, axis=1)
    
    # Center the points
    X = points1t - pb[:, np.newaxis]
    Y = points2t - qb[:, np.newaxis]
    
    # Compute covariance matrix
    S = X @ Y.T
    
    # Perform SVD
    U, _, Vt = np.linalg.svd(S)
    V = Vt.T
    
    # Ensure proper rotation matrix (determinant = +1)
    sigma = np.eye(U.shape[1])
    if np.linalg.det(V @ U.T) < 0:
        sigma[-1, -1] = -1
    
    # Compute rotation and translation
    R = V @ sigma @ U.T
    t = qb - R @ pb
    
    return R, t


def transform_points(points, R, t):
    """
    Transform points using rotation matrix R and translation vector t.
    
    Args:
        points: Points to transform (N x 3 numpy array)
        R: Rotation matrix (3 x 3)
        t: Translation vector (3,)
    
    Returns:
        Transformed points (N x 3 numpy array)
    """
    points_t = points.T
    points_transformed_t = points_t - t[:, np.newaxis]
    points_transformed_t = R.T @ points_transformed_t
    return points_transformed_t.T


def compute_point_distances(points1, points2):
    """
    Compute Euclidean distances between corresponding points.
    
    Args:
        points1, points2: Nx3 arrays of corresponding points
    
    Returns:
        Array of distances
    """
    return np.sqrt(np.sum((points1 - points2) ** 2, axis=1))


def calculate_seamline_statistics(distances):
    """
    Calculate comprehensive statistics for stretch coefficients.
    
    Parameters:
        optim_weft, optim_warp: Optimized stretch coefficients
        target_weft, target_warp: Target stretch coefficients
    
    Returns:
        Dictionary containing various statistics for both u and v directions
    """
    return {
        'mean': np.mean(distances),
        'median': np.median(distances),
        'max': np.max(distances),
        'min': np.min(distances),
        'std': np.std(distances),
        'percentile_95': np.percentile(distances, 95),
        'percentile_99': np.percentile(distances, 99),
        'box_plot_stats': compute_box_plot_stats(distances)  # Added box plot stats
    }


def get_seamline_statistics():
    """
    Process all seamline files in the directory.

    Returns:
        Dictionary mapping seamline names to arrays of point-to-point distances
    """
    seamlines_dir = 'data/seamlines/'
    optim_2ds = []
    for patch_label in PATCH_LIST:
        optim_2ds.append(trimesh.load(
            os.path.join('results/pattern/latest/', patch_label, 'optim_final-seams.ply')))
    
    _distances_dict = {}
    statistics_dict = {}
    
    for filename in os.listdir(seamlines_dir):
        seamline_name = os.path.splitext(filename)[0]
            
        # Get patch indices and names
        patch_idx1, patch_idx2 = SEAM_TO_PATCH_PAIRS[seamline_name]
        patch_name1 = ID_TO_PATCH[patch_idx1]
        patch_name2 = ID_TO_PATCH[patch_idx2]
        
        # Get corresponding meshes
        mesh1 = optim_2ds[patch_name1]
        mesh2 = optim_2ds[patch_name2]
        
        # Read seamline indices
        seam_path = os.path.join(seamlines_dir, filename)
        idx1, idx2 = read_seamline(seam_path)
        
        # Extract seamline vertices
        points1 = mesh1.vertices[idx1]
        points2 = mesh2.vertices[idx2]
        
        # Compute Procrustes alignment
        R, t = procrustes_alignment(points1, points2)
        
        # Transform second set of points
        points2_aligned = transform_points(points2, R, t)
        
        # Compute distances between corresponding points
        distances = compute_point_distances(points1, points2_aligned)
        
        _distances_dict[seamline_name] = distances
        statistics_dict[seamline_name] = calculate_seamline_statistics(distances)
        
    _distances_dict['all'] = np.concatenate(distances)
    statistics_dict['all'] = calculate_seamline_statistics(_distances_dict['all'])
    
    return statistics_dict
