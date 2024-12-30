import os
import trimesh
import numpy as np
from scipy import stats


def load_stretch_data(method, patch_label):
    bary_rootdir = 'data/bary/ref_2d'
    mesh_rootdir = 'data/embedded/'
    param_2d_rootdir = 'data/param_2d/'
    scales_rootdir = 'data/scales/'
    
    bary_subdir = os.path.join(bary_rootdir, method, patch_label)
    for bary_fname in [x for x in os.listdir(bary_subdir) if 'final-seams' in x]:
        bary_fpath = os.path.join(bary_subdir, bary_fname)
        suffix = bary_fpath.split('.')[0].split('_')[-1]
        if '_u_' in bary_fpath:
            DU_bary = np.loadtxt(bary_fpath)
        else:
            DV_bary = np.loadtxt(bary_fpath)
    
    mesh_3d_path = os.path.join(mesh_rootdir, patch_label, 'ref.ply')
    param_2d_path = os.path.join(param_2d_rootdir, patch_label, 'optim_final-seams.ply')
    scales_u_path = os.path.join(scales_rootdir, patch_label, 'scales_u.txt')
    scales_v_path = os.path.join(scales_rootdir, patch_label, 'scales_v.txt')
    
    ref_mesh_3d = trimesh.load(mesh_3d_path)
    ref_param_2d = trimesh.load(param_2d_path)
    target_scales_u = np.loadtxt(scales_u_path)
    target_scales_v = np.loadtxt(scales_v_path)
    
    optim_u, optim_v, target_u, target_v = extract_stretches(
        V_2d=ref_param_2d.vertices, 
        V_3d=ref_mesh_3d.vertices, 
        F=ref_mesh_3d.faces, 
        DU_bary=DU_bary, 
        DV_bary=DV_bary, 
        target_scale_vec_u=target_scales_u, 
        target_scale_vec_v=target_scales_v
    )
    
    return optim_u, optim_v, target_u, target_v


def extract_stretches(V_2d, V_3d, F, DU_bary, DV_bary, target_scale_vec_u, target_scale_vec_v):
    """
    Extract optimized and target stretches from barycentric coordinates.
    
    Parameters:
        V_2d: numpy array of shape (n_vertices, 2) - 2D vertices
        V_3d: numpy array of shape (n_vertices, 3) - 3D vertices
        F: numpy array of shape (n_faces, 3) - faces
        DU_bary: numpy array of shape (n_faces, 3) - U barycentric coordinates
        DV_bary: numpy array of shape (n_faces, 3) - V barycentric coordinates
        target_scale_vec_u: numpy array of shape (n_faces,) - U target scales
        target_scale_vec_v: numpy array of shape (n_faces,) - V target scales
    
    Returns:
        optim_vec_u: numpy array of shape (n_faces,) - optimized U stretches
        optim_vec_v: numpy array of shape (n_faces,) - optimized V stretches
        target_vec_u: numpy array of shape (n_faces,) - target U stretches
        target_vec_v: numpy array of shape (n_faces,) - target V stretches
    """
    n_faces = F.shape[0]
    optim_vec_u = np.zeros(n_faces)
    optim_vec_v = np.zeros(n_faces)
    target_vec_u = np.zeros(n_faces)
    target_vec_v = np.zeros(n_faces)
    
    # Common barycentric coordinates for centroid
    D_bary = np.array([1/3, 1/3, 1/3])
    
    for f_id in range(n_faces):
        # Get vertices of current face
        face_verts_2d = V_2d[F[f_id]]
        face_verts_3d = V_3d[F[f_id]]
        
        # Compute 3D positions using barycentric coordinates
        Dp = np.sum(D_bary[:, np.newaxis] * face_verts_3d, axis=0)
        DUp = np.sum(DU_bary[f_id][:, np.newaxis] * face_verts_3d, axis=0)
        DVp = np.sum(DV_bary[f_id][:, np.newaxis] * face_verts_3d, axis=0)
        
        # Compute target vectors
        target_vec_u[f_id] = np.linalg.norm(DUp - Dp) * target_scale_vec_u[f_id]
        target_vec_v[f_id] = np.linalg.norm(DVp - Dp) * target_scale_vec_v[f_id]
        
        # Compute optimized vectors
        optim_vec_u[f_id] = np.sum(face_verts_2d[:, 0] * (DU_bary[f_id] - D_bary))
        optim_vec_v[f_id] = np.sum(face_verts_2d[:, 1] * (DV_bary[f_id] - D_bary))
    
    return optim_vec_u, optim_vec_v, target_vec_u, target_vec_v


def calculate_stretch_statistics(optim_u, optim_v, target_u, target_v):
    """
    Calculate comprehensive statistics for stretch coefficients.
    
    Parameters:
        optim_u, optim_v: Optimized stretch coefficients
        target_u, target_v: Target stretch coefficients
    
    Returns:
        Dictionary containing various statistics for both u and v directions
    """
    def compute_directional_stats(y, t):
        # Absolute differences
        abs_diff = np.abs(y - t)
        # Relative differences (y/t)
        rel_diff = y / t
        # L2 norm of differences
        f2_norm = np.sqrt(np.sum((y - t)**2))
        # Relative error percentage
        rel_error = np.abs(y - t) / t * 100
        
        stats_dict = {
            'abs_diff': {
                'mean': np.mean(abs_diff),
                'median': np.median(abs_diff),
                'max': np.max(abs_diff),
                'min': np.min(abs_diff),
                'std': np.std(abs_diff),
                'percentile_95': np.percentile(abs_diff, 95),
                'percentile_99': np.percentile(abs_diff, 99)
            },
            'rel_diff': {
                'mean': np.mean(rel_diff),
                'median': np.median(rel_diff),
                'max': np.max(rel_diff),
                'min': np.min(rel_diff),
                'std': np.std(rel_diff)
            },
            'f2_norm': f2_norm,
            'rel_error': {
                'mean': np.mean(rel_error),
                'median': np.median(rel_error),
                'max': np.max(rel_error),
                'min': np.min(rel_error),
                'std': np.std(rel_error)
            },
            'additional_metrics': {
                'rmse': np.sqrt(np.mean((y - t)**2)),  # Root Mean Square Error
                'mae': np.mean(np.abs(y - t)),         # Mean Absolute Error
                'r2_score': stats.pearsonr(y, t)[0]**2,  # R-squared correlation
                'skewness': stats.skew(abs_diff),      # Skewness of absolute differences
                'kurtosis': stats.kurtosis(abs_diff)   # Kurtosis of absolute differences
            },
            'raw_stats': {
                'optim': {
                    'mean': np.mean(y),
                    'median': np.median(y),
                    'std': np.std(y),
                    'max': np.max(y),
                    'min': np.min(y)
                },
                'target': {
                    'mean': np.mean(t),
                    'median': np.median(t),
                    'std': np.std(t),
                    'max': np.max(t),
                    'min': np.min(t)
                }
            }
        }
        return stats_dict

    # Compute statistics for both directions
    stats_dict = {
        'u_direction': compute_directional_stats(optim_u, target_u),
        'v_direction': compute_directional_stats(optim_v, target_v),
        'global_metrics': {
            'total_f2_norm': np.sqrt(np.sum((optim_u - target_u)**2) + np.sum((optim_v - target_v)**2)),
            'combined_rmse': np.sqrt(np.mean((optim_u - target_u)**2) + np.mean((optim_v - target_v)**2)),
            'mean_relative_error': np.mean(np.concatenate([
                np.abs(optim_u - target_u) / target_u * 100,
                np.abs(optim_v - target_v) / target_v * 100
            ]))
        }
    }
    
    return stats_dict

# Example usage:
def print_stretch_statistics(stats_dict, precision=4):
    """
    Helper function to print the statistics in a readable format.
    """
    for direction in ['u_direction', 'v_direction']:
        print(f"\n=== {direction.upper()} ===")
        
        # Print absolute difference statistics
        print("\nAbsolute Differences (|y-t|):")
        for key, value in stats_dict[direction]['abs_diff'].items():
            print(f"{key:15}: {value:.{precision}f}")
            
        # Print relative difference statistics
        print("\nRelative Differences (y/t):")
        for key, value in stats_dict[direction]['rel_diff'].items():
            print(f"{key:15}: {value:.{precision}f}")
            
        # Print F2 norm
        print(f"\nF2 Norm: {stats_dict[direction]['f2_norm']:.{precision}f}")
        
        # Print additional metrics
        print("\nAdditional Metrics:")
        for key, value in stats_dict[direction]['additional_metrics'].items():
            print(f"{key:15}: {value:.{precision}f}")
    
    # Print global metrics
    print("\n=== GLOBAL METRICS ===")
    for key, value in stats_dict['global_metrics'].items():
        print(f"{key:20}: {value:.{precision}f}")
        
        
def _map_stretch_to_color(stretch, min_stretch=0.7, max_stretch=1.3):
    normalized_stretch = (stretch - min_stretch) / (max_stretch - min_stretch)
    intensity = int((1 - normalized_stretch) * 255)
    color = np.array([intensity, intensity, 255 - intensity, 255], dtype=np.uint8)
    return color


def color_code_stretches(verts, faces, stretch_array, min_stretch=0.7, max_stretch=1.3):
    # Ensure the stretch array length matches the number of faces
    assert len(stretch_array) == len(faces), "The length of stretch_array must match the number of faces."

    # Initialize vertex colors
    vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)

    # Count occurrences of each vertex in faces to average the colors
    vertex_counts = np.zeros(verts.shape[0], dtype=np.int32)

    # Apply the color coding
    for face, stretch in zip(faces, stretch_array):
        color = _map_stretch_to_color(stretch, min_stretch, max_stretch)
        for vertex in face:
            vertex_colors[vertex] += color
            vertex_counts[vertex] += 1

    # Average the colors for each vertex
    for i in range(len(vertex_colors)):
        if vertex_counts[i] > 0:
            vertex_colors[i] //= vertex_counts[i]
        else:
            vertex_colors[i] = [128, 128, 128, 255]  # Default gray color if no face uses this vertex

    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)
