import numpy as np
import trimesh

from tailorlang.eval.stretch_utils import color_code_stretches


def extract_submesh(vertices, faces, face_fractions_dict):
    """
    Extract a submesh given the original mesh and a list of face indices.
    
    Parameters:
    vertices: np.array of shape (N, 3) containing vertex coordinates
    faces: np.array of shape (M, 3) containing face vertex indices
    face_fractions_dict: dict of {face index: fraction included} to extract
    
    Returns:
    new_vertices: np.array of vertices in the submesh
    new_faces: np.array of faces with updated vertex indices
    vertex_map: mapping from original vertex indices to new ones
    """
    # Get selected faces
    selected_faces = faces[list(face_fractions_dict.keys())]
    
    # Get unique vertices used in selected faces
    unique_vertices = np.unique(selected_faces.flatten())
    
    # Create vertex mapping from old to new indices
    vertex_map = {old: new for new, old in enumerate(unique_vertices)}
    
    # Create new vertex array
    new_vertices = vertices[unique_vertices]
    
    # Update face indices to use new vertex indices
    new_faces = np.array([[vertex_map[idx] for idx in face] 
                         for face in selected_faces])
    
    return new_vertices, new_faces, vertex_map


def find_edge_points(vertices, faces, left_dart_cut_idx, right_dart_cut_idx, distance, dart_side):
    """
    Find two points on the mesh edge by traversing in both directions until reaching
    or exceeding the specified distance.
    
    Parameters:
    vertices: np.ndarray (V, 3) - Mesh vertices
    faces: np.ndarray (F, 3) - Mesh faces (indices)
    center_vertex_idx: int - Index of the center vertex on the edge
    distance: float - Desired geodesic distance from center to each point
    
    Returns:
    tuple: (left_point_idx, right_point_idx)
    """
    # Build vertex adjacency for edge vertices
    V = len(vertices)
    vertex_face_count = np.zeros(V)
    vertex_neighbors = {i: set() for i in range(V)}
    
    # Count faces per vertex and build adjacency
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            vertex_face_count[v1] += 1
            vertex_neighbors[v1].add(v2)
            vertex_neighbors[v2].add(v1)
    
    # Identify edge vertices (vertices with fewer faces)
    edge_vertices = set(np.where(vertex_face_count < 6)[0])
    
    def get_edge_directions(vertex_idx):
        """Get neighbors of a vertex that are also on the edge"""
        return [n for n in vertex_neighbors[vertex_idx] if n in edge_vertices]
    
    def traverse_edge(start_vertex, neighbor):
        """
        Traverse edge from start_vertex through initial_neighbor until reaching
        specified distance or end of edge
        """
        current = neighbor
        prev = start_vertex
        accumulated_distance = np.linalg.norm(vertices[current] - vertices[start_vertex])
        path = [start_vertex, current]
        
        while accumulated_distance < distance:
            # Get edge neighbors excluding the one we came from
            next_vertices = [v for v in get_edge_directions(current) if v != prev]
            
            if not next_vertices:  # Reached end of edge
                break
                
            # Choose the neighbor that makes the path most straight
            # (closest to current direction)
            if len(next_vertices) > 1:
                current_dir = vertices[current] - vertices[prev]
                current_dir = current_dir / np.linalg.norm(current_dir)
                
                max_alignment = -1
                best_next = next_vertices[0]
                
                for next_v in next_vertices:
                    next_dir = vertices[next_v] - vertices[current]
                    next_dir = next_dir / np.linalg.norm(next_dir)
                    alignment = np.dot(current_dir, next_dir)
                    
                    if alignment > max_alignment:
                        max_alignment = alignment
                        best_next = next_v
                
                next_vertex = best_next
            else:
                next_vertex = next_vertices[0]
            
            prev = current
            current = next_vertex
            accumulated_distance += np.linalg.norm(vertices[current] - vertices[prev])
            path.append(current)
        
        return current, accumulated_distance, path
    
    # Get initial edge neighbors for center vertex
    left_edge_directions = get_edge_directions(left_dart_cut_idx)
    right_edge_directions = get_edge_directions(right_dart_cut_idx)
    
    # Traverse in both directions
    if dart_side == 'right':
        left_point, left_dist, left_path = traverse_edge(left_dart_cut_idx, left_edge_directions[1])
        right_point, right_dist, right_path = traverse_edge(right_dart_cut_idx, right_edge_directions[1])
    else:
        left_point, left_dist, left_path = traverse_edge(left_dart_cut_idx, left_edge_directions[1])
        right_point, right_dist, right_path = traverse_edge(right_dart_cut_idx, right_edge_directions[0])
    
    return left_point, right_point


def triangle_area(A, B, C):
    # Convert points to numpy arrays for easier calculation
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # Calculate two vectors from the three points
    v = B - A  # Vector from A to B
    w = C - A  # Vector from A to C
    
    # Calculate cross product
    cross_product = np.cross(v, w)
    
    # Calculate magnitude of cross product
    area = 0.5 * np.linalg.norm(cross_product)
    
    return area


def select_faces_in_dart(vertices, faces, left_dart_cut_idx, right_dart_cut_idx, dart_tip_idx, dart_size, max_distance_from_plane, dart_side):
    """
    Select faces within a dart-shaped region on a mesh and calculate overlap fractions.
    Excludes faces that are too far from the reference plane.
    
    Parameters:
    vertices: np.ndarray (V, 3) - Mesh vertices
    faces: np.ndarray (F, 3) - Mesh faces (indices)
    edge_vertex_idx: int - Index of the vertex on the edge
    inner_vertex_idx: int - Index of the inner vertex
    dart_size: float - Size of the dart (geodesic distance between edge points)
    max_distance_from_plane: float - Maximum allowed distance from the reference plane
    
    Returns:
    dict: Dictionary mapping face indices to their overlap fractions (relative to face area)
    """
    # Step 1: Find edge vertices (same as before)
    left_triangle_idx, right_triangle_idx = find_edge_points(
        vertices, faces, left_dart_cut_idx, right_dart_cut_idx, dart_size/2, dart_side
    )
    
    # Step 2: Create reference plane and define distance checking
    p1 = vertices[left_triangle_idx]
    p2 = vertices[right_triangle_idx]
    p3 = vertices[dart_tip_idx]
    
    # Define plane using three points
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # Point-plane distance function
    def distance_to_plane(point):
        """Calculate signed distance from a point to the reference plane"""
        return abs(np.dot(point - p1, normal))
    
    def is_face_near_plane(face_vertices):
        """Check if face is within threshold distance of the reference plane"""
        face_distances = [distance_to_plane(v) for v in face_vertices]
        return max(face_distances) <= max_distance_from_plane
    
    def project_point(point):
        v = point - p1
        dist = np.dot(v, normal)
        return point - dist * normal
    
    proj_p1 = project_point(p1)
    proj_p2 = project_point(p2)
    proj_p3 = project_point(p3)
    
    def polygon_intersection_area(poly1_vertices, poly2_vertices):
        """
        Calculate approximate intersection area using Monte Carlo sampling.
        Returns the ratio of intersection area to poly1 area.
        """
        min_x = min(min(v[0] for v in poly1_vertices), min(v[0] for v in poly2_vertices))
        max_x = max(max(v[0] for v in poly1_vertices), max(v[0] for v in poly2_vertices))
        min_y = min(min(v[1] for v in poly1_vertices), min(v[1] for v in poly2_vertices))
        max_y = max(max(v[1] for v in poly1_vertices), max(v[1] for v in poly2_vertices))
        
        def point_in_polygon(point, vertices):
            x, y = point
            inside = False
            j = len(vertices) - 1
            for i in range(len(vertices)):
                if ((vertices[i][1] > y) != (vertices[j][1] > y) and
                    x < (vertices[j][0] - vertices[i][0]) * (y - vertices[i][1]) /
                    (vertices[j][1] - vertices[i][1]) + vertices[i][0]):
                    inside = not inside
                j = i
            return inside
        
        num_samples = 1000
        points_in_intersection = 0
        points_in_poly1 = 0
        
        for _ in range(num_samples):
            x = min_x + (max_x - min_x) * np.random.random()
            y = min_y + (max_y - min_y) * np.random.random()
            point = (x, y)
            
            in_poly1 = point_in_polygon(point, poly1_vertices)
            if in_poly1:
                points_in_poly1 += 1
                if point_in_polygon(point, poly2_vertices):
                    points_in_intersection += 1
        
        if points_in_poly1 == 0:
            return 0.0
            
        return points_in_intersection / points_in_poly1
    
    # Step 3: Select faces and calculate overlap fractions
    face_fractions = {}
    dart_vertices = [proj_p1[:2], proj_p2[:2], proj_p3[:2]]
    
    for face_idx, face in enumerate(faces):
        # Get face vertices
        face_vertices = [vertices[v_idx] for v_idx in face]
        
        # First check if face is near enough to the plane
        if not is_face_near_plane(face_vertices):
            continue
            
        # If face is near enough, proceed with projection and intersection test
        proj_face_vertices = [project_point(v)[:2] for v in face_vertices]
        
        # Calculate intersection fraction relative to face area
        overlap_fraction = polygon_intersection_area(proj_face_vertices, dart_vertices)
        
        if overlap_fraction > 0:
            face_fractions[face_idx] = overlap_fraction
    
    return face_fractions, triangle_area(p1, p2, p3)


if __name__ == '__main__':
    upper_front_mesh = trimesh.load_mesh('data/embedded/ui/upper_front/ref.ply')
    orig_uniform_coef = 1.1
    stretch_values = np.array([orig_uniform_coef] * upper_front_mesh.faces.shape[0], dtype=np.float32)
    
    # NOTE: For now, not using the fractions, only the "full" faces
    inner_face_fractions_dict, inner_triangle_area = select_faces_in_dart(
        vertices=upper_front_mesh.vertices, 
        faces=upper_front_mesh.faces, 
        left_dart_cut_idx=561,
        right_dart_cut_idx=749, 
        dart_tip_idx=752, 
        dart_size=0.05,
        max_distance_from_plane=0.1,
        dart_side='left'
    )
    print(inner_face_fractions_dict)
    inner_subverts, inner_subfaces, _ = extract_submesh(vertices=upper_front_mesh.vertices, faces=upper_front_mesh.faces, face_fractions_dict=inner_face_fractions_dict)
    trimesh.Trimesh(vertices=inner_subverts, faces=inner_subfaces).export('inner_submesh.ply')
    
    # NOTE: For now, not using the fractions, only the "full" faces
    outer_face_fractions_dict, outer_triangle_area = select_faces_in_dart(
        vertices=upper_front_mesh.vertices, 
        faces=upper_front_mesh.faces, 
        left_dart_cut_idx=561,
        right_dart_cut_idx=749, 
        dart_tip_idx=752,
        dart_size=0.1,
        max_distance_from_plane=0.1,
        dart_side='left'
    )
    print(outer_face_fractions_dict)
    outer_subverts, outer_subfaces, _ = extract_submesh(vertices=upper_front_mesh.vertices, faces=upper_front_mesh.faces, face_fractions_dict=outer_face_fractions_dict)
    trimesh.Trimesh(vertices=outer_subverts, faces=outer_subfaces).export('outer_submesh.ply')
    
    # NOTE: The new coefficient is simply a ratio between the difference between the larger and smaller triangle divided by the larger times original coefficient
    # TODO: Later, perhaps, use face fractions to more precisely update the coefficients
    new_coef = ((outer_triangle_area - inner_triangle_area) / outer_triangle_area) * orig_uniform_coef
    for face_idx in outer_face_fractions_dict:
        stretch_values[face_idx] = new_coef
        
    colored_upper_front_mesh = color_code_stretches(
        verts=upper_front_mesh.vertices,
        faces=upper_front_mesh.faces,
        stretch_array=stretch_values,
        min_stretch=stretch_values.min(),
        max_stretch=stretch_values.max()
    )
    colored_upper_front_mesh.export('dart_stretches.ply')
