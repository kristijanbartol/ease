import os
import cv2
import trimesh
import numpy as np
from scipy import stats

from tailorlang.const import PATCH_LIST
from tailorlang.eval.const import GLOBAL_IMG_SCALE
from tailorlang.eval.box_plot_utils import compute_box_plot_stats


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


def extract_stretch_ratios(V_2d, V_3d_ref, F, DU_bary, DV_bary):
    """
    Extract stretch ratios between 2D and 3D triangles, i.e., the scales.
    
    Parameters:
        V_2d: numpy array of shape (n_vertices, 2) - 2D vertices
        V_3d: numpy array of shape (n_vertices, 3) - 3D vertices
        F: numpy array of shape (n_faces, 3) - faces
        DU_bary: numpy array of shape (n_faces, 3) - U barycentric coordinates
        DV_bary: numpy array of shape (n_faces, 3) - V barycentric coordinates
    
    Returns:
        scales_u: numpy array of shape (n_faces,) - optimized U stretches
        scales_v: numpy array of shape (n_faces,) - optimized V stretches
    """
    n_faces = F.shape[0]
    scales_u = np.zeros(n_faces)
    scales_v = np.zeros(n_faces)
    ref_stretch_u = np.zeros(n_faces)
    ref_stretch_v = np.zeros(n_faces)
    optim_vec_u = np.zeros(n_faces)
    optim_vec_v = np.zeros(n_faces)
    
    # Common barycentric coordinates for centroid
    D_bary = np.array([1/3, 1/3, 1/3])
    
    for f_id in range(n_faces):
        # Get vertices of current face
        face_verts_2d = V_2d[F[f_id]]
        face_verts_3d = V_3d_ref[F[f_id]]
        
        # Compute 3D positions using barycentric coordinates
        Dp = np.sum(D_bary[:, np.newaxis] * face_verts_3d, axis=0)
        DUp = np.sum(DU_bary[f_id][:, np.newaxis] * face_verts_3d, axis=0)
        DVp = np.sum(DV_bary[f_id][:, np.newaxis] * face_verts_3d, axis=0)
        
        # Compute target vectors
        ref_stretch_u[f_id] = np.linalg.norm(DUp - Dp)
        ref_stretch_v[f_id] = np.linalg.norm(DVp - Dp)
        
        # Compute optimized vectors
        optim_vec_u[f_id] = np.sum(face_verts_2d[:, 0] * (DU_bary[f_id] - D_bary))
        optim_vec_v[f_id] = np.sum(face_verts_2d[:, 1] * (DV_bary[f_id] - D_bary))
        
        scales_u[f_id] = optim_vec_u[f_id] / ref_stretch_u[f_id]
        scales_v[f_id] = optim_vec_v[f_id] / ref_stretch_v[f_id]
    
    return scales_u, scales_v


def extract_stretch_ratios_with_target(V_2d, V_3d_ref, V_3d_target, F, DU_bary, DV_bary):
    """
    Extract stretch ratios between 2D and 3D triangles, i.e., the scales.
    
    Parameters:
        V_2d: numpy array of shape (n_vertices, 2) - 2D vertices
        V_3d: numpy array of shape (n_vertices, 3) - 3D vertices
        F: numpy array of shape (n_faces, 3) - faces
        DU_bary: numpy array of shape (n_faces, 3) - U barycentric coordinates
        DV_bary: numpy array of shape (n_faces, 3) - V barycentric coordinates
    
    Returns:
        scales_u: numpy array of shape (n_faces,) - optimized U stretches
        scales_v: numpy array of shape (n_faces,) - optimized V stretches
    """
    n_faces = F.shape[0]
    scales_u = np.zeros(n_faces)
    scales_v = np.zeros(n_faces)
    ref_stretch_u = np.zeros(n_faces)
    ref_stretch_v = np.zeros(n_faces)
    target_stretch_u = np.zeros(n_faces)
    target_stretch_v = np.zeros(n_faces)
    final_stretch_u = np.zeros(n_faces)
    final_stretch_v = np.zeros(n_faces)
    optim_vec_u = np.zeros(n_faces)
    optim_vec_v = np.zeros(n_faces)
    
    # Common barycentric coordinates for centroid
    D_bary = np.array([1/3, 1/3, 1/3])
    
    for f_id in range(n_faces):
        # Get vertices of current face
        face_verts_2d = V_2d[F[f_id]]
        face_verts_3d_ref = V_3d_ref[F[f_id]]
        face_verts_3d_target = V_3d_target[F[f_id]]
        
        # Compute 3D positions using barycentric coordinates - Reference
        Dp_ref = np.sum(D_bary[:, np.newaxis] * face_verts_3d_ref, axis=0)
        DUp_ref = np.sum(DU_bary[f_id][:, np.newaxis] * face_verts_3d_ref, axis=0)
        DVp_ref = np.sum(DV_bary[f_id][:, np.newaxis] * face_verts_3d_ref, axis=0)
        
        # Compute 3D positions using barycentric coordinates - Target
        Dp_target = np.sum(D_bary[:, np.newaxis] * face_verts_3d_target, axis=0)
        DUp_target = np.sum(DU_bary[f_id][:, np.newaxis] * face_verts_3d_target, axis=0)
        DVp_target = np.sum(DV_bary[f_id][:, np.newaxis] * face_verts_3d_target, axis=0)
        
        # Compute target vectors - Reference
        ref_stretch_u[f_id] = np.linalg.norm(DUp_ref - Dp_ref)
        ref_stretch_v[f_id] = np.linalg.norm(DVp_ref - Dp_ref)
        
        # Compute target vectors - Target
        target_stretch_u[f_id] = np.linalg.norm(DUp_target - Dp_target)
        target_stretch_v[f_id] = np.linalg.norm(DVp_target - Dp_target)
        
        # Calculate final stretch targets
        final_stretch_u[f_id] = max(target_stretch_u[f_id], ref_stretch_u[f_id])
        final_stretch_v[f_id] = max(target_stretch_v[f_id], ref_stretch_v[f_id])
        
        # Compute optimized vectors
        optim_vec_u[f_id] = np.sum(face_verts_2d[:, 0] * (DU_bary[f_id] - D_bary))
        optim_vec_v[f_id] = np.sum(face_verts_2d[:, 1] * (DV_bary[f_id] - D_bary))
        
        scales_u[f_id] = optim_vec_u[f_id] / final_stretch_u[f_id]
        scales_v[f_id] = optim_vec_v[f_id] / final_stretch_v[f_id]
    
    return scales_u, scales_v


def load_stretch_data(patch_label):
    bary_rootdir = 'data/bary/ref_2d'
    mesh_rootdir = 'data/embedded/'
    param_2d_rootdir = 'data/param_2d/'
    scales_rootdir = 'data/scales/'
    
    bary_subdir = os.path.join(bary_rootdir, patch_label)
    for bary_fname in [x for x in os.listdir(bary_subdir) if 'final-seams' in x]:
        bary_fpath = os.path.join(bary_subdir, bary_fname)
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
    
    if 'sleeve' in patch_label:
        optim_weft, optim_warp, target_weft, target_warp = optim_v, optim_u, target_v, target_u
    else:
        optim_weft, optim_warp, target_weft, target_warp = optim_u, optim_v, target_u, target_v
    
    return optim_weft, optim_warp, target_weft, target_warp


def get_stretch_statistics():
    """
    Analyze stretch statistics for individual patches and combined data.
    
    Returns:
        Dictionary containing statistics for each patch and combined statistics
    """
    patch_statistics = {}
    all_optim_weft, all_optim_warp = [], []
    all_target_weft, all_target_warp = [], []
    
    # Collect statistics for individual patches
    for patch_label in PATCH_LIST:
        # Load stretch data for current patch
        optim_weft, optim_warp, target_weft, target_warp = load_stretch_data(patch_label)
        
        # Calculate statistics for current patch
        patch_statistics[patch_label], _ = calculate_stretch_statistics(
            optim_weft=optim_weft,
            optim_warp=optim_warp,
            target_weft=target_weft,
            target_warp=target_warp
        )
        print(f"\nStatistics for patch: {patch_label}")
        print_stretch_statistics(patch_statistics[patch_label])
        
        # Collect data for combined analysis
        all_optim_weft.append(optim_weft)
        all_optim_warp.append(optim_warp)
        all_target_weft.append(target_weft)
        all_target_warp.append(target_warp)
    
    # Combine all arrays using numpy.concatenate
    combined_optim_weft = np.concatenate(all_optim_weft)
    combined_optim_warp = np.concatenate(all_optim_warp)
    combined_target_weft = np.concatenate(all_target_weft)
    combined_target_warp = np.concatenate(all_target_warp)
    
    
    # TODO: Concatenate also upper and lower patches separately (for 3D simulation visualization)
    
    
    # Calculate statistics for combined data
    patch_statistics['all'], rel_weft_ratios = calculate_stretch_statistics(
        optim_weft=combined_optim_weft,
        optim_warp=combined_optim_warp,
        target_weft=combined_target_weft,
        target_warp=combined_target_warp
    )
    
    print("\nCombined statistics across all patches:")
    print_stretch_statistics(patch_statistics['all'])
    
    return patch_statistics


def calculate_stretch_statistics(optim_weft, optim_warp, target_weft, target_warp):
    """
    Calculate comprehensive statistics for stretch coefficients.
    
    Parameters:
        optim_weft, optim_warp: Optimized stretch coefficients
        target_weft, target_warp: Target stretch coefficients
    
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
                'percentile_99': np.percentile(abs_diff, 99),
                'box_plot_stats': compute_box_plot_stats(abs_diff)  # Added box plot stats
            },
            'rel_diff': {
                'mean': np.mean(rel_diff),
                'median': np.median(rel_diff),
                'max': np.max(rel_diff),
                'min': np.min(rel_diff),
                'std': np.std(rel_diff),
                'box_plot_stats': compute_box_plot_stats(rel_diff)  # Added box plot stats
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
                    'min': np.min(y)
                }
            }
        }
        return stats_dict

    # Compute statistics for both directions
    stats_dict = {
        'weft_direction': compute_directional_stats(optim_weft, target_weft),
        'warp_direction': compute_directional_stats(optim_warp, target_warp),
        'global_metrics': {
            'total_f2_norm': np.sqrt(np.sum((optim_weft - target_weft)**2) + np.sum((optim_warp - target_warp)**2)),
            'combined_rmse': np.sqrt(np.mean((optim_weft - target_weft)**2) + np.mean((optim_warp - target_warp)**2)),
            'mean_relative_error': np.mean(np.concatenate([
                np.abs(optim_weft - target_weft) / target_weft * 100,
                np.abs(optim_warp - target_warp) / target_warp * 100
            ])),
            # Add box plot stats for combined data
            'combined_abs_diff_stats': compute_box_plot_stats(
                np.concatenate([np.abs(optim_weft - target_weft), np.abs(optim_warp - target_warp)])
            ),
            'combined_rel_diff_stats': compute_box_plot_stats(
                np.concatenate([optim_weft / target_weft, optim_warp / target_warp])
            )
        }
    }
    
    return stats_dict, optim_weft / target_weft

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
        
        
def _map_garment_stretch_to_color(stretch, tightness_max=0.3, looseness_max=0.4):
    if stretch > 1.0:
        # Blue range for loose areas (darker blue = more loose)
        normalized_stretch = min(1.0, (stretch - 1.0) / looseness_max)
        intensity = int(normalized_stretch * 255)
        return np.array([255 - intensity, 255 - intensity, 255, 255], dtype=np.uint8)
    else:
        # Red range for tight areas (brighter red = more tight)
        normalized_stretch = min(1.0, (1.0 - stretch) / tightness_max)
        intensity = int(normalized_stretch * 255)
        return np.array([255, 255 - intensity, 255 - intensity, 255], dtype=np.uint8)
    
    
def _map_garment_stretch_to_alternative_scale(stretch, tightness_max=0.3, looseness_max=0.4):
    if stretch > 1.0:
        # Blue range for loose areas (darker blue = more loose)
        normalized_stretch = min(1.0, (stretch - 1.0) / looseness_max)
        intensity = int(normalized_stretch * 255)
        return np.array([255 - intensity, 255, 255 - intensity, 255], dtype=np.uint8)
    else:
        # Red range for tight areas (brighter red = more tight)
        normalized_stretch = min(1.0, (1.0 - stretch) / tightness_max)
        intensity = int(normalized_stretch * 255)
        return np.array([255, 255 - intensity, 255, 255], dtype=np.uint8)


'''
def color_code_stretches(verts, faces, stretch_array, tightness_max=0.15, looseness_max=0.4):
    # Ensure the stretch array length matches the number of faces
    assert len(stretch_array) == len(faces), "The length of stretch_array must match the number of faces."

    # Initialize vertex colors
    vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)

    # Count occurrences of each vertex in faces to average the colors
    vertex_counts = np.zeros(verts.shape[0], dtype=np.int32)

    # Apply the color coding
    for face, stretch in zip(faces, stretch_array):
        color = _map_garment_stretch_to_color(stretch, tightness_max, looseness_max)
        for vertex in face:
            vertex_colors[vertex] += color
            vertex_counts[vertex] += 1

    # Average the colors for each vertex
    for i in range(len(vertex_colors)):
        if vertex_counts[i] > 0:
            vertex_colors[i] //= vertex_counts[i]
        else:
            vertex_colors[i] = [128, 128, 128, 255]  # Default gray color if no face uses this vertex

    return vertex_colors
'''


def color_code_stretches(verts, faces, stretch_array):
    assert len(stretch_array) == len(faces), "The length of stretch_array must match the number of faces."
    
    # Initialize arrays for accumulating colors and counts
    vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.float32)  # Using float32 for accumulation
    vertex_counts = np.zeros(verts.shape[0], dtype=np.int32)
    
    # First pass: accumulate stretch values per vertex
    vertex_stretches = np.zeros(verts.shape[0], dtype=np.float32)
    for face, stretch in zip(faces, stretch_array):
        for vertex in face:
            vertex_stretches[vertex] += stretch
            vertex_counts[vertex] += 1
    
    # Average the stretch values
    mask = vertex_counts > 0
    vertex_stretches[mask] /= vertex_counts[mask]
    
    # Apply color mapping to averaged stretch values
    for i in range(len(vertex_colors)):
        if vertex_counts[i] > 0:
            vertex_colors[i] = _map_garment_stretch_to_color(vertex_stretches[i])
        else:
            vertex_colors[i] = [128, 128, 128, 255]  # Default gray
            
    return vertex_colors.astype(np.uint8)


'''
def _map_subdivided_to_original_faces(orig_verts, orig_faces, subdiv_verts, subdiv_faces):
    """
    Create a mapping from subdivided faces to their original faces.
    
    Returns:
    --------
    np.ndarray
        Array where index i contains the index of the original face that contained
        subdivided face i
    """
    # Calculate centroids of original faces
    orig_centroids = np.array([np.mean(orig_verts[face], axis=0) for face in orig_faces])
    
    # Calculate centroids of subdivided faces
    subdiv_centroids = np.array([np.mean(subdiv_verts[face], axis=0) for face in subdiv_faces])
    
    # For each subdivided face, find the closest original face centroid
    mapping = np.zeros(len(subdiv_faces), dtype=np.int32)
    
    for i, subdiv_centroid in enumerate(subdiv_centroids):
        # Calculate distances to all original centroids
        distances = np.linalg.norm(orig_centroids - subdiv_centroid, axis=1)
        # Find the closest original face
        mapping[i] = np.argmin(distances)
    
    return mapping


def color_code_stretches_subdivided(orig_verts, orig_faces, subdiv_verts, subdiv_faces, stretch_array, min_stretch=0.7, max_stretch=1.3):
    """
    Calculate vertex colors for a subdivided mesh based on stretch values from the original mesh.
    
    Parameters:
    -----------
    orig_verts : np.ndarray
        Vertices of the original mesh
    orig_faces : np.ndarray
        Faces of the original mesh
    subdiv_verts : np.ndarray
        Vertices of the subdivided mesh
    subdiv_faces : np.ndarray
        Faces of the subdivided mesh
    stretch_array : np.ndarray
        Stretch values for each face in the original mesh
    min_stretch : float, optional
        Minimum stretch value for color mapping (default: 0.7)
    max_stretch : float, optional
        Maximum stretch value for color mapping (default: 1.3)
        
    Returns:
    --------
    np.ndarray
        Array of RGBA colors for each vertex in the subdivided mesh
    """
    # Ensure the stretch array length matches the number of original faces
    assert len(stretch_array) == len(orig_faces), "The length of stretch_array must match the number of original faces."
    
    # Initialize vertex colors for subdivided mesh
    vertex_colors = np.zeros((len(subdiv_verts), 4), dtype=np.uint8)
    vertex_counts = np.zeros(len(subdiv_verts), dtype=np.int32)
    
    # Create a mapping from subdivided faces to original faces
    subdiv_to_orig = _map_subdivided_to_original_faces(orig_verts, orig_faces, subdiv_verts, subdiv_faces)
    
    # Apply the color coding
    for subdiv_face_idx, subdiv_face in enumerate(subdiv_faces):
        # Get the original face index and its stretch value
        orig_face_idx = subdiv_to_orig[subdiv_face_idx]
        stretch = stretch_array[orig_face_idx]
        
        # Map stretch to color
        color = _map_garment_stretch_to_color(stretch, min_stretch, max_stretch)
        
        # Apply color to vertices of the subdivided face
        for vertex in subdiv_face:
            vertex_colors[vertex] += color
            vertex_counts[vertex] += 1
    
    # Average the colors for each vertex
    for i in range(len(vertex_colors)):
        if vertex_counts[i] > 0:
            vertex_colors[i] //= vertex_counts[i]
        else:
            vertex_colors[i] = [128, 128, 128, 255]  # Default gray color
    
    return vertex_colors
'''


def calculate_vertex_colors(mesh, face_stretches, min_stretch=0.7, max_stretch=1.3):
    """
    Calculate smoothed vertex colors based on adjacent face stretch values.
    
    Args:
        mesh: Trimesh object containing the mesh
        face_stretches: Array of stretch values per face
        min_stretch/max_stretch: Range for color mapping
    
    Returns:
        Array of colors for each vertex
    """
    vertex_stretches = np.zeros(len(mesh.vertices))
    vertex_counts = np.zeros(len(mesh.vertices))
    
    # Accumulate stretch values from adjacent faces
    for face_idx, face in enumerate(mesh.faces):
        stretch = face_stretches[face_idx]
        vertex_stretches[face] += stretch
        vertex_counts[face] += 1
    
    # Average the stretch values
    vertex_stretches = np.divide(
        vertex_stretches, 
        vertex_counts, 
        out=np.zeros_like(vertex_stretches), 
        where=vertex_counts != 0
    )
    
    # Map to colors
    vertex_colors = np.array([
        _map_garment_stretch_to_color(s, min_stretch, max_stretch) 
        for s in vertex_stretches
    ])
    
    return vertex_colors

def interpolate_color(p, triangle_points, vertex_colors):
    """
    Interpolate color at point p using barycentric coordinates.
    
    Args:
        p: Point to interpolate color at
        triangle_points: Vertices of the triangle
        vertex_colors: Colors at the triangle vertices
    
    Returns:
        Interpolated color at point p
    """
    # Calculate barycentric coordinates
    v0 = triangle_points[1] - triangle_points[0]
    v1 = triangle_points[2] - triangle_points[0]
    v2 = p - triangle_points[0]
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    # Interpolate colors using barycentric coordinates
    color = u * vertex_colors[0] + v * vertex_colors[1] + w * vertex_colors[2]
    return color.astype(np.uint8)

def mesh_to_stretch_image(mesh, face_stretches, image_size=(800, 800), 
                         min_stretch=0.85, max_stretch=1.15):
    """
    Create a smoothly interpolated visualization of stretch values.
    
    Args:
        mesh: Trimesh object containing the mesh
        face_stretches: Array of stretch values per face
        image_size: Size of output image
        min_stretch/max_stretch: Range for color mapping
    
    Returns:
        Image array with smooth stretch visualization
    """
    # Project vertices to 2D as in your original code
    points = np.array(mesh.vertices)
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    points[:, :2] = (points[:, :2] - min_bounds[:2]) * GLOBAL_IMG_SCALE
    
    # Calculate vertex colors
    vertex_colors = calculate_vertex_colors(mesh, face_stretches, min_stretch, max_stretch)
    
    # Create output image
    image = np.zeros((*image_size, 4), dtype=np.uint8)
    
    # For each triangle
    for face_idx, face in enumerate(mesh.faces):
        # Get 2D coordinates of triangle vertices
        tri_points = points[face, :2].astype(int)
        
        # Get bounding box of triangle
        min_x = max(tri_points[:, 0].min(), 0)
        max_x = min(tri_points[:, 0].max() + 1, image_size[0])
        min_y = max(tri_points[:, 1].min(), 0)
        max_y = min(tri_points[:, 1].max() + 1, image_size[1])
        
        # For each pixel in bounding box
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                point = np.array([x, y])
                
                # Check if point is inside triangle
                if cv2.pointPolygonTest(tri_points, (x, y), False) >= 0:
                    # Interpolate color
                    color = interpolate_color(
                        point, 
                        tri_points, 
                        vertex_colors[face]
                    )
                    image[y, x] = color
    
    return image[::-1]
