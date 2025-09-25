import bpy
import bmesh
import argparse
import sys
import os

if 'bpy' in sys.modules and not '--' in sys.argv:
    project_dir = '/Users/kristijanbartol/LOOM/' if sys.platform == 'darwin' else '/home/kristijan/LOOM/'

    # Paths to the meshes
    body_path = os.path.join(project_dir, 'data/body/target-00_x10.ply')
    shirt_path = os.path.join(project_dir, 'results/non-skintight/sit-pose_long_bezier_1_2.0_1.0_2.0FFF/base_upper.ply')
    pant_path = os.path.join(project_dir, 'results/non-skintight/sit-pose_long_bezier_1_2.0_1.0_2.0FFF/base_lower.ply')
    body_output_path = os.path.join(project_dir, 'results/sim/body.ply')
    shirt_output_path = os.path.join(project_dir, 'results/sim/shirt.ply')
    pant_output_path = os.path.join(project_dir, 'results/sim/pant.ply')
    is_dress = False
    is_skirt = False
    shoulderless = False
else:
    # Get command line arguments after "--"
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--body', required=True, help='Path to body mesh')
    parser.add_argument('--shirt', required=True, help='Path to shirt mesh')
    parser.add_argument('--pant', required=True, help='Path to pant mesh')
    parser.add_argument('--body-output', required=True, help='Output path for body mesh')
    parser.add_argument('--shirt-output', required=True, help='Output path for shirt mesh')
    parser.add_argument('--pant-output', required=True, help='Output path for pant mesh')
    parser.add_argument('--is-dress', action='store_true', help='Flag for dress (do not simulate pants)')
    parser.add_argument('--is-skirt', action='store_true', help='Flag for skirt (use a softer lower garment material)')
    parser.add_argument('--shoulderless', action='store_true', help='Flag for shoulderless upper design (needs pinning)')

    args = parser.parse_args(argv)
    
    body_path = args.body
    shirt_path = args.shirt
    pant_path = args.pant
    body_output_path = args.body_output
    shirt_output_path = args.shirt_output
    pant_output_path = args.pant_output
    is_dress = args.is_dress
    is_skirt = args.is_skirt
    shoulderless = args.shoulderless


# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)


### Process pant object ###
if not is_dress:

    # Import the lower mesh
    bpy.ops.wm.ply_import(filepath=pant_path)
    pant = bpy.context.selected_objects[0]
    pant.name = "Pant"

    pant_object = bpy.data.objects.get("Pant")

    # Create a new vertex group for pinning
    pin_group = pant_object.vertex_groups.new(name="Pin")

    # Create BMesh for better topology analysis
    bm = bmesh.new()
    bm.from_mesh(pant_object.data)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Find boundary edges
    boundary_edges = [e for e in bm.edges if e.is_boundary]

    # Find the vertices that form the upper boundary
    # First, get all boundary vertices
    boundary_verts = set()
    for edge in boundary_edges:
        boundary_verts.update(v.index for v in edge.verts)

    # Then filter for upper portion (e.g., top 10% of the mesh height)
    z_coords = [v.co.z for v in bm.verts]
    max_z = max(z_coords)
    min_z = min(z_coords)
    height_threshold = max_z - (max_z - min_z) * 0.1  # Adjust 0.1 to change how much of the top to consider

    waistline_verts = [v.index for v in bm.verts if v.index in boundary_verts and v.co.z > height_threshold]

    # Add vertices to the pin group with weight 1.0
    pin_group.add(waistline_verts, 1.0, 'REPLACE')

    # Free the BMesh
    bm.free()

    # Add the cloth modifier
    pant_cloth_modifier = pant_object.modifiers.new(name="Cloth", type='CLOTH')

    # Set the pin group in cloth physics
    pant_cloth_modifier.settings.vertex_group_mass = "Pin"

    # Configure cloth physics settings
    if is_skirt:
        pant_cloth_modifier.settings.quality = 18
        pant_cloth_modifier.settings.mass = 0.45
        pant_cloth_modifier.settings.air_damping = 1.0

        pant_cloth_modifier.settings.tension_stiffness = 5
        pant_cloth_modifier.settings.compression_stiffness = 5
        pant_cloth_modifier.settings.shear_stiffness = 5
        pant_cloth_modifier.settings.bending_stiffness = 0.03

        pant_cloth_modifier.settings.tension_damping = 3
        pant_cloth_modifier.settings.compression_damping = 3
        pant_cloth_modifier.settings.shear_damping = 3
        pant_cloth_modifier.settings.bending_damping = 0.08
    else:
        pant_cloth_modifier.settings.quality = 12
        pant_cloth_modifier.settings.mass = 1
        pant_cloth_modifier.settings.air_damping = 1
        pant_cloth_modifier.settings.tension_stiffness = 40
        pant_cloth_modifier.settings.compression_stiffness = 40
        pant_cloth_modifier.settings.shear_stiffness = 40
        pant_cloth_modifier.settings.bending_stiffness = 10

        pant_cloth_modifier.settings.tension_damping = 25
        pant_cloth_modifier.settings.compression_damping = 25
        pant_cloth_modifier.settings.shear_damping = 25
        pant_cloth_modifier.settings.bending_damping = 0.5

    pant_cloth_modifier.collision_settings.use_collision = True
    pant_cloth_modifier.collision_settings.distance_min = 0.003

### Process shirt object ###

# Import the upper mesh
bpy.ops.wm.ply_import(filepath=shirt_path)
shirt = bpy.context.selected_objects[0]
shirt.name = "Shirt"

shirt_object = bpy.data.objects.get("Shirt")

# Add the cloth modifier
shirt_cloth_modifier = shirt_object.modifiers.new(name="Cloth", type='CLOTH')

# Ako je shoulderless, pinaj gornjih 10% graničnih vrhova (svijetni Z, objekt nije rotiran)
if shoulderless:
    shirt_pin_group = shirt_object.vertex_groups.get("Pin") or shirt_object.vertex_groups.new(name="Pin")

    bm = bmesh.new()
    bm.from_mesh(shirt_object.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    # skupi boundary vrhove
    boundary_verts = set()
    for e in bm.edges:
        if e.is_boundary:
            boundary_verts.add(e.verts[0].index)
            boundary_verts.add(e.verts[1].index)

    if boundary_verts:
        M = shirt_object.matrix_world  # world Z
        b_sorted = sorted(boundary_verts, key=lambda i: (M @ bm.verts[i].co).z, reverse=True)

        frac = 0.25  # gornjih 25%
        k = max(12, int(len(b_sorted) * frac))  # minimalan broj da se uhvati "prsten"
        top_boundary_strip = b_sorted[:k]

        shirt_pin_group.add(top_boundary_strip, 1.0, 'REPLACE')

        # poveži pin grupu s clothom
        shirt_cloth_modifier.settings.vertex_group_mass = "Pin"

    bm.free()

# Configure cloth physics settings

if False:
    shirt_cloth_modifier.settings.quality = 18
    shirt_cloth_modifier.settings.mass = 0.45
    shirt_cloth_modifier.settings.air_damping = 1.0

    shirt_cloth_modifier.settings.tension_stiffness = 5
    shirt_cloth_modifier.settings.compression_stiffness = 5
    shirt_cloth_modifier.settings.shear_stiffness = 5
    shirt_cloth_modifier.settings.bending_stiffness = 0.03

    shirt_cloth_modifier.settings.tension_damping = 3
    shirt_cloth_modifier.settings.compression_damping = 3
    shirt_cloth_modifier.settings.shear_damping = 3
    shirt_cloth_modifier.settings.bending_damping = 0.08

    '''
    if is_dress:
        shirt_cloth_modifier.settings.quality = 16
        shirt_cloth_modifier.settings.mass = 0.75
        shirt_cloth_modifier.settings.air_damping = 0.4

        shirt_cloth_modifier.settings.tension_stiffness = 8
        shirt_cloth_modifier.settings.compression_stiffness = 8
        shirt_cloth_modifier.settings.shear_stiffness = 8
        shirt_cloth_modifier.settings.bending_stiffness = 0.08

        shirt_cloth_modifier.settings.tension_damping = 6
        shirt_cloth_modifier.settings.compression_damping = 6
        shirt_cloth_modifier.settings.shear_damping = 6
        shirt_cloth_modifier.settings.bending_damping = 0.15
    '''

else:
    shirt_cloth_modifier.settings.quality = 5
    shirt_cloth_modifier.settings.mass = 0.6
    shirt_cloth_modifier.settings.air_damping = 1
    shirt_cloth_modifier.settings.tension_stiffness = 15
    shirt_cloth_modifier.settings.compression_stiffness = 15
    shirt_cloth_modifier.settings.shear_stiffness = 15
    shirt_cloth_modifier.settings.bending_stiffness = 0.5

    shirt_cloth_modifier.settings.tension_damping = 5
    shirt_cloth_modifier.settings.compression_damping = 5
    shirt_cloth_modifier.settings.shear_damping = 5
    shirt_cloth_modifier.settings.bending_damping = 0.5

shirt_cloth_modifier.collision_settings.use_collision = True
#shirt_cloth_modifier.collision_settings.distance_min = 0.08        # only for long shirt + loose pants
shirt_cloth_modifier.collision_settings.distance_min = 0.06        # only for long shirt + loose pants
#shirt_cloth_modifier.collision_settings.distance_min = 0.025


### Process body mesh ###

# Import the body mesh
bpy.ops.wm.ply_import(filepath=body_path)
body = bpy.context.selected_objects[0]
body.name = "Body"

# Enable collisions for the body
bpy.context.view_layer.objects.active = body
bpy.ops.object.modifier_add(type='COLLISION')

# Set up timeline
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 50
bpy.context.scene.frame_current = 1

# Run the simulation frame by frame
print("Starting simulation...")
for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    print(f"Simulating frame {frame}/{bpy.context.scene.frame_end}")

print("Simulation complete")

### Export the individual meshes ###

# Export body
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.wm.ply_export(filepath=body_output_path, export_selected_objects=True)
print(f"Exported body mesh to: {body_output_path}")

# Export shirt
bpy.ops.object.select_all(action='DESELECT')
shirt.select_set(True)
bpy.context.view_layer.objects.active = shirt
bpy.ops.wm.ply_export(filepath=shirt_output_path, export_selected_objects=True)
print(f"Exported simulated shirt to: {shirt_output_path}")

# Export pant
if not is_dress:
    bpy.ops.object.select_all(action='DESELECT')
    pant.select_set(True)
    bpy.context.view_layer.objects.active = pant
    bpy.ops.wm.ply_export(filepath=pant_output_path, export_selected_objects=True)
    print(f"Exported simulated shirt to: {pant_output_path}")
