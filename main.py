import numpy as np
from smplx import SMPL
import matplotlib.pyplot as plt


RIGHT_ARMPIT = [
    4767, 4766, 4892, 4891, 6412, 4132, 4131, 4134, 4106, 4109, 6300, 4959, 4962,
    4804, 4118, 4121, 4165, 4332, 6378, 5259, 5258, 4423, 4311, 4927, 4801, 4712,
    4334, 4335, 4390, 4459, 4460, 4466, 4465, 4570, 4519, 4496, 4495, 4539, 4556,
    4559
]
LEFT_ARMPIT = [
    1285, 1286, 1418, 1419, 2953, 644, 645, 646, 618, 619, 2839, 1487, 1488, 1323, 
    630, 631, 677, 846, 2919, 1795, 1796, 937, 822, 823, 1454, 1321, 1228, 850, 849, 
    905, 973, 974, 980, 979, 1022, 1033, 1009, 1010, 1052, 1070, 1072
]
RIGHT_SHOULDER = [6469, 5322, 5325, 4721, 4724, 4270, 4198, 4094, 4097, 4230, 4306, 4305]
LEFT_SHOULDER = [3010, 1861, 1862, 1238, 1239, 783, 711, 606, 607, 742, 818, 817]
RIGHT_OUTER_PANT = [6378, 5259, 5258, 4423, 4311, 4310, 4927, 4801, 4712, 4334, 4335, 4390, 4459, 4460, 
                    4466, 4465, 4507, 4519, 4496, 4495, 4539, 4556, 4559, 4585, 4586, 4943, 
                    4603, 4604, 4622, 6589, 6590, 6608, 6869]
LEFT_OUTER_PANT = [2919, 1795, 1796, 937, 822, 823, 1454, 1321, 1228, 850, 849, 905, 973, 974, 980, 979, 
                   1022, 1033, 1009, 1010, 1052, 1070, 1072, 1101, 1100, 1470, 1118, 1117, 
                   1136, 3189, 3190, 3208, 3469]
RIGHT_INNER_PANT = [1208, 4364, 4367, 4925, 6553, 4709, 4708, 4450, 4449, 4435, 4438, 4439, 
                    4442, 4514, 4527, 4502, 4501, 4543, 4542, 4565, 4996, 4574, 4573, 4591, 
                    4590, 4609, 6577, 6576, 6598, 6833]
LEFT_INNER_PANT = [1208, 878, 879, 1451, 3131, 1226, 1227, 963, 964, 949, 950, 953, 954, 1028, 
                   1042, 1014, 1015, 1059, 1056, 1078, 1525, 1088, 1089, 1107, 1104, 1122, 
                   3175, 3176, 3198, 3433]

SEAM_IDX_DICT = {
    'upper_front': {
        'right_armpit': RIGHT_ARMPIT, 
        'left_armpit':  LEFT_ARMPIT,
        'right_arm':    [4767, 5008, 4685, 4163, 4162, 5359, 6467, 6468, 6469],
        'left_arm':     [1285, 1537, 1199, 673, 674, 1898, 3008, 3009, 3010],
        'neck':         [4305, 4308, 4309, 4787, 4788, 4058, 4059, 3060, 573, 570, 1308, 1307, 821, 819, 817],
        'right_shoulder': RIGHT_SHOULDER,
        'left_shoulder':  LEFT_SHOULDER
    },
    'upper_back': {
        'right_armpit': RIGHT_ARMPIT, 
        'left_armpit':  LEFT_ARMPIT,
        'right_arm':    [6470, 6352, 6351, 5346, 5345, 5352, 6462, 6459, 6402, 5302, 4200, 4203, 5340, 4243, 4242, 4769, 4768, 4767, 6469],
        'left_arm':     [3011, 2893, 2894, 1887, 1884, 1891, 3003, 3000, 2943, 1841, 712, 713, 1879, 755, 756, 1288, 1284, 1285],
        'neck':         [817, 803, 806, 813, 812, 1219, 3470, 3470, 4702, 4302, 4301, 4292, 4291, 4305],
        'right_shoulder': RIGHT_SHOULDER,
        'left_shoulder':  LEFT_SHOULDER
    },
    'sleeves': {
        'right': [6469, 4124, 4125, 5321, 4872, 4873, 4880, 4881, 4791, 4790, 5148, 5108, 5111, 5122, 5125, 5082, 5085, 
                  5376, 5211, 5161, 5160, 5028, 5025, 5412, 5413, 5387, 5386, 5570, 5480],
        'left':  [3010, 636, 635, 1860, 1400, 1399, 1407, 1406, 1311, 1310, 1679, 1639, 1640, 1653, 1654, 1613, 1614, 1915, 
                  1742, 1691, 1692, 1557, 1556, 1951, 1950, 1925, 1926, 2109, 2019]
    },
    'pant_front_right': {
        'right_outer': RIGHT_OUTER_PANT,
        'right_inner': RIGHT_INNER_PANT,
        'mid_inner': [3507, 3160, 1806, 1807, 3510, 3145, 3146, 3148, 3149, 1208],
        'waistline': [6378, 6383, 6385, 6386, 6379, 6380, 6384, 6381, 6382, 3507]
    },
    'pant_front_left': {
        'left_outer' : LEFT_OUTER_PANT,
        'left_inner' : LEFT_INNER_PANT,
        'mid_inner': [3507, 3160, 1806, 1807, 3510, 3145, 3146, 3148, 3149, 1208],
        'waistline': [3507, 2922, 2923, 2925, 2920, 2921, 2926, 2927, 2924, 2919]
    },
    'pant_back_right': {
        'right_outer': RIGHT_OUTER_PANT,
        'right_inner': RIGHT_INNER_PANT,
        'mid_inner': [1784, 1783, 3159, 3158, 3484, 3120, 3119, 3141, 3481, 3102, 1540, 1539, 1476, 1364, 1363, 1515, 3172, 3170, 1353, 1278, 1210, 1209, 1208],
        'waistline': [1784, 5246, 5247, 6544, 6371, 6370, 6376, 6377, 6374, 6375, 6378]
    },
    'pant_back_left': {
        'left_outer' : LEFT_OUTER_PANT,
        'left_inner' : LEFT_INNER_PANT,
        'mid_inner': [1784, 1783, 3159, 3158, 3484, 3120, 3119, 3141, 3481, 3102, 1540, 1539, 1476, 1364, 1363, 1515, 3172, 3170, 1353, 1278, 1210, 1209, 1208],
        'waistline': [2919, 2915, 2916, 2917, 2918, 2911, 2910, 3122, 1780, 1781, 1784]
    }
}
INIT_UPPER_FRONT = 4173
INIT_UPPER_BACK = 4238
INIT_RIGHT_SLEEVE = 5119
INIT_LEFT_SLEEVE = 788
INIT_RIGHT_FRONT_PANT = 4952
INIT_LEFT_FRONT_PANT = 1479
INIT_RIGHT_BACK_PANT = 4339
INIT_LEFT_BACK_PANT = 897

SHIRT_LENGTH = 0.25
SLEEVE_LENGTH = 0.4
PANT_LENGTH = 0.5


COLOR_MAP = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'light_green': (144, 238, 144),
    'dark_green': (0, 100, 0),
    'orange': (255, 165, 0),
    # Add other colors as needed
}


def visualize_all_verts(
        verts, 
        front_vertex_indices, 
        back_vertex_indices, 
        right_sleeve_indices, 
        left_sleeve_indices,
        pant_front_right_indices,
        pant_front_left_indices,
        pant_back_right_indices,
        pant_back_left_indices
    ):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize all vertices as gray
    vertex_colors = ['gray'] * len(verts)

    # Color the front vertices red, the back vertices blue, the right sleeve dark green, and the left sleeve light green
    for i in front_vertex_indices:
        vertex_colors[i] = 'red'
    for i in back_vertex_indices:
        vertex_colors[i] = 'blue'
    for i in right_sleeve_indices:
        vertex_colors[i] = 'darkgreen'
    for i in left_sleeve_indices:
        vertex_colors[i] = 'lightgreen'
    for i in pant_front_right_indices:
        vertex_colors[i] = 'darkblue'
    for i in pant_front_left_indices:
        vertex_colors[i] = 'lightblue'
    for i in pant_back_right_indices:
        vertex_colors[i] = 'orange'
    for i in pant_back_left_indices:
        vertex_colors[i] = 'brown'

    # Scatter plot for vertices
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c=vertex_colors)

    # Set the labels for the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Show the plot to the screen
    plt.show(block=True)


def visualize_verts(verts, vertex_indices, color):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize all vertices as gray
    vertex_colors = ['gray'] * len(verts)

    # Color the selected vertices with the given color
    for i in vertex_indices:
        vertex_colors[i] = color

    # Scatter plot for vertices
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c=vertex_colors)

    # Set the labels for the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Show the plot to the screen
    plt.show(block=True)


def extract_parameterized_seams(verts, garment_length, seam_vertex_indices, inner_seams=False):
    remaining_length = garment_length
    last_point = None

    for i in range(len(seam_vertex_indices) - 1):
        start_vertex = verts[seam_vertex_indices[i]]
        end_vertex = verts[seam_vertex_indices[i + 1]]
        
        edge_length = np.linalg.norm(end_vertex - start_vertex)

        if edge_length < remaining_length:
            remaining_length -= edge_length
        else:
            # Find the point along the edge that corresponds to the remaining length
            direction = (end_vertex - start_vertex) / edge_length
            last_point = start_vertex + direction * remaining_length
            # Break the loop as we have reached the specified length
            break
    
    return seam_vertex_indices[:(i + 1) + 1], last_point


def determine_shirt_seams(verts, shirt_length, seam_idx_dict):
    right_armpit_points, last_right_point = extract_parameterized_seams(verts, shirt_length, seam_idx_dict['right_armpit'])
    left_armpit_points, _ = extract_parameterized_seams(verts, shirt_length, seam_idx_dict['left_armpit'])
    shirt_seams = right_armpit_points + \
                  seam_idx_dict['right_arm'] + \
                  seam_idx_dict['right_shoulder'] + \
                  seam_idx_dict['neck'] + \
                  seam_idx_dict['left_shoulder'] + \
                  seam_idx_dict['left_arm'] + \
                  left_armpit_points
    return shirt_seams, last_right_point[1]


def determine_pant_seams(verts, pant_length, seam_idx_dict, side, inner_seams=True):
    pant_seams = []
    outer_points, last_outer_point = extract_parameterized_seams(verts, pant_length, seam_idx_dict[f'{side}_outer'])
    pant_seams += outer_points
    if inner_seams:
        inner_points, _ = extract_parameterized_seams(verts, pant_length, seam_idx_dict[f'{side}_inner'])
        pant_seams += inner_points + seam_idx_dict['mid_inner']
    pant_seams += seam_idx_dict['waistline']
    return pant_seams, last_outer_point[1]


def build_vertex_adjacency_list(F):
    adjacency_list = {}
    for triangle in F:
        for vertex in triangle:
            if vertex not in adjacency_list:
                adjacency_list[vertex] = set()
            # Add the neighboring vertices
            adjacency_list[vertex].update(triangle)
            adjacency_list[vertex].remove(vertex)  # A vertex is not a neighbor to itself
    return adjacency_list


def flood_fill_vertices(vertex_positions, adjacency_list, boundary_vertices, y_threshold, start_vertex):
    # Convert boundary vertices to a set for efficient lookup
    boundary_set = set(boundary_vertices)

    # Initialize the stack with the start vertex, which is assumed to be inside the boundary and above the y_threshold
    stack = [start_vertex]

    # Initialize the set to keep track of visited vertices
    visited = set()

    # Initialize the set to store the selected vertices
    selected_vertices = set()

    while stack:
        # Pop the last vertex from the stack
        vertex_idx = stack.pop()

        # If the vertex has not been visited yet and is not on the boundary or below y_threshold, process it
        if vertex_idx not in visited and vertex_idx not in boundary_set and vertex_positions[vertex_idx, 1] >= y_threshold:
            # Mark the vertex as visited and add to selected
            visited.add(vertex_idx)
            selected_vertices.add(vertex_idx)

            # Iterate over the neighbors of the current vertex
            for neighbor_idx in adjacency_list[vertex_idx]:
                # If the neighbor hasn't been visited, add it to the stack
                if neighbor_idx not in visited:
                    stack.append(neighbor_idx)

            #if len(selected_vertices) % 10 == 0 and start_vertex == INIT_RIGHT_FRONT_PANT:
            #    visualize_verts(vertex_positions, selected_vertices, color='red')
    
    # Once all possible vertices have been visited, combine the selected vertices with the boundary vertices
    # Note that only those boundary vertices whose Y coordinate is larger than a threshold should be selected
    thresh_boundaries = [index for index in boundary_vertices if vertex_positions[index][1] > y_threshold]
    selected_vertices.update(thresh_boundaries)

    return list(selected_vertices)


def select_sleeve_verts(verts, adjacency_list, start_vertex_index, seam_indices, sleeve_length, x_direction_multiplier):
    # Get the X coordinates of the starting and ending seam vertices
    x_start = verts[seam_indices[0], 0]

    # Define the range of X values based on the sleeve length and the direction multiplier
    if x_direction_multiplier == -1:
        # For the right sleeve, we find vertices backwards from the start
        x_min = x_start - sleeve_length
        x_max = x_start
    else:
        # For the left sleeve, we find vertices forwards from the start
        x_min = x_start
        x_max = x_start + sleeve_length

    # Ensure x_min is less than x_max
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)

    # Initialize the set for selected vertex indices
    selected_indices = set()
    
    # List to keep track of the vertices that need to be visited
    to_visit = [start_vertex_index]
    
    while to_visit:
        current_index = to_visit.pop()
        if current_index not in selected_indices and x_min <= verts[current_index, 0] <= x_max:
            selected_indices.add(current_index)
            # Get the indices of the vertices connected to the current vertex
            connected_vertices = adjacency_list[current_index]
            # Add unvisited vertices to the to_visit list
            for vertex_index in connected_vertices:
                if vertex_index not in selected_indices:
                    to_visit.append(vertex_index)
    
    return list(selected_indices)


def compute_vertex_normals(verts, faces):
    # Calculate the normals for each face
    face_normals = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                            verts[faces[:, 2]] - verts[faces[:, 0]])
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, np.newaxis]

    # Initialize vertex normals as zero
    vertex_normals = np.zeros_like(verts)

    # Add face normals to each vertex normal
    for i, face in enumerate(faces):
        vertex_normals[face] += face_normals[i]

    # Normalize the vertex normals
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]

    return vertex_normals


def apply_offset_to_verts(verts, faces, offset):
    # Compute the vertex normals
    vertex_normals = compute_vertex_normals(verts, faces)

    # Apply the offset to each vertex along its normal
    offset_verts = verts + vertex_normals * offset

    return offset_verts


def extract_garment_mesh(verts, faces, garment_vertex_indices, offset=0):
    # Apply offset if specified
    if offset != 0:
        verts = apply_offset_to_verts(verts, faces, offset)

    # Extract vertices and faces for the garment
    garment_verts = verts[garment_vertex_indices]
    garment_faces = [face for face in faces if set(face).issubset(garment_vertex_indices)]

    # Create a mapping from the original vertex indices to the new, local indices
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(garment_vertex_indices)}
    # Update the indices in the faces to the new local indices
    garment_faces = np.array([[index_mapping[v] for v in face] for face in garment_faces])

    return garment_verts, garment_faces


def write_ply_file(filename, verts, faces, vertex_colors):
    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write vertex list with colors
        color_mapping = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'light_green': (128, 255, 128),
            'dark_green': (0, 128, 0),
            'orange': (255, 128, 0),
            'dark_blue': (0, 0, 139),  # Dark blue color
            'light_blue': (173, 216, 230),  # Light blue color, also known as baby blue
            'brown': (165, 42, 42),  # Assuming you also need brown color defined
            'yellow': (255, 255, 0)  # Yellow color
        }
        
        for vert in verts:
            color = vertex_colors.get(tuple(vert), 'gray')  # Default color is gray
            r, g, b = color_mapping.get(color, (128, 128, 128))  # Convert color name to RGB
            f.write(f"{vert[0]} {vert[1]} {vert[2]} {r} {g} {b}\n")
        
        # Write face list
        for face in faces:
            f.write(f"3 {' '.join(str(v) for v in face)}\n")


def export_garments_to_ply(verts, faces, vertex_indices_by_color, filename_prefix):
    # Map each vertex to its color
    vertex_colors = {tuple(verts[i]): color for color, indices in vertex_indices_by_color.items() for i in indices}
    
    # Make sure faces are referring to the correct vertex indices
    used_vertex_indices = set(idx for indices in vertex_indices_by_color.values() for idx in indices)
    verts = verts[list(used_vertex_indices)]
    
    # Update the face indices
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(sorted(used_vertex_indices))}
    updated_faces = [[index_mapping[idx] for idx in face] for face in faces if set(face).issubset(used_vertex_indices)]

    # Write to PLY
    ply_filename = f"{filename_prefix}.ply"
    write_ply_file(ply_filename, verts, updated_faces, vertex_colors)


def update_color_indices(garment_vertex_indices, color_dict):
    # Create a mapping from old SMPL indices to new garment mesh indices
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(garment_vertex_indices)}
    
    # Update the color dictionary with the new indices
    new_color_dict = {}
    for color, indices in color_dict.items():
        # Update the indices using the mapping, filter out indices not found in the mapping
        new_indices = [index_mapping.get(index) for index in indices if index in index_mapping]
        # Remove any None values that were not in the mapping
        new_indices = [index for index in new_indices if index is not None]
        new_color_dict[color] = new_indices
    
    return new_color_dict


if __name__ == '__main__':
    smpl_model = SMPL(model_path='/data/hierprob3d/smpl/SMPL_FEMALE.pkl')
    verts = smpl_model().vertices[0].cpu().detach().numpy()
    faces = smpl_model.faces

    # Build adjacency list for vertices
    adjacency_list = build_vertex_adjacency_list(faces)

    # For the front shirt
    seam_idxs_front, y_shirt_threshold = determine_shirt_seams(verts, SHIRT_LENGTH, SEAM_IDX_DICT['upper_front'])
    front_v_idxs = flood_fill_vertices(verts, adjacency_list, seam_idxs_front, y_shirt_threshold, INIT_UPPER_FRONT)
    
    # For the back shirt
    seam_idxs_back, _ = determine_shirt_seams(verts, SHIRT_LENGTH, SEAM_IDX_DICT['upper_back'])
    back_v_idxs = flood_fill_vertices(verts, adjacency_list, seam_idxs_back, y_shirt_threshold, INIT_UPPER_BACK)

    # Selection and visualization of the sleeves
    # For the right sleeve
    right_sleeve_v_idxs = select_sleeve_verts(verts, adjacency_list, INIT_RIGHT_SLEEVE, SEAM_IDX_DICT['sleeves']['right'], SLEEVE_LENGTH, -1)

    # For the left sleeve
    left_sleeve_v_idxs = select_sleeve_verts(verts, adjacency_list, INIT_LEFT_SLEEVE, SEAM_IDX_DICT['sleeves']['left'], SLEEVE_LENGTH, 1)

    # For the front right pant
    seams_pant_front_right, y_pant_threshold = determine_pant_seams(verts, PANT_LENGTH, SEAM_IDX_DICT['pant_front_right'], 'right')
    pant_front_right_v_idxs = flood_fill_vertices(verts, adjacency_list, seams_pant_front_right, y_pant_threshold, INIT_RIGHT_FRONT_PANT)
    
    # For the front left pant
    seams_pant_front_left, _ = determine_pant_seams(verts, PANT_LENGTH, SEAM_IDX_DICT['pant_front_left'], 'left')
    pant_front_left_v_idxs = flood_fill_vertices(verts, adjacency_list, seams_pant_front_left, y_pant_threshold, INIT_LEFT_FRONT_PANT)

    # For the back right pant
    seams_pant_back_right, _ = determine_pant_seams(verts, PANT_LENGTH, SEAM_IDX_DICT['pant_back_right'], 'right')
    pant_back_right_v_idxs = flood_fill_vertices(verts, adjacency_list, seams_pant_back_right, y_pant_threshold, INIT_RIGHT_BACK_PANT)
    
    # For the back left pant
    seams_pant_back_left, _ = determine_pant_seams(verts, PANT_LENGTH, SEAM_IDX_DICT['pant_back_left'], 'left')
    pant_back_left_v_idxs = flood_fill_vertices(verts, adjacency_list, seams_pant_back_left, y_pant_threshold, INIT_LEFT_BACK_PANT)

    # Create upper and lower garment meshes
    offset_distance = 0.01
    upper_indices = front_v_idxs + back_v_idxs + right_sleeve_v_idxs + left_sleeve_v_idxs
    lower_indices = pant_front_right_v_idxs + pant_front_left_v_idxs + pant_back_right_v_idxs + pant_back_left_v_idxs
    upper_garment_verts, upper_garment_faces = extract_garment_mesh(verts, faces, upper_indices, offset=offset_distance)
    lower_garment_verts, lower_garment_faces = extract_garment_mesh(verts, faces, lower_indices, offset=offset_distance)

    # Export garment meshes with color
    upper_garment_colors = {
        'red': front_v_idxs,
        'blue': back_v_idxs,
        'light_green': right_sleeve_v_idxs,
        'dark_green': left_sleeve_v_idxs
    }
    lower_garment_colors = {
        'dark_blue': pant_front_right_v_idxs,
        'light_blue': pant_front_left_v_idxs,
        'orange': pant_back_right_v_idxs,
        'yellow': pant_back_left_v_idxs
    }
    body_colors = {
        'gray': list(range(len(verts)))
    }
    updated_upper_garment_colors = update_color_indices(upper_indices, upper_garment_colors)
    updated_lower_garment_colors = update_color_indices(lower_indices, lower_garment_colors)
    export_garments_to_ply(upper_garment_verts, upper_garment_faces, updated_upper_garment_colors, 'upper_garment')
    export_garments_to_ply(lower_garment_verts, lower_garment_faces, updated_lower_garment_colors, 'lower_garment')
    export_garments_to_ply(verts, faces, body_colors, 'body')
