import matplotlib.pyplot as plt


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
