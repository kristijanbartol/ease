import bpy

# Paths to the meshes
body_path = "/Users/kristijanbartol/TailorLang/data/body/ref.ply"
garment_path = "/Users/kristijanbartol/TailorLang/upper_deformed.ply"
output_path = "/Users/kristijanbartol/TailorLang/simulated.ply"

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import the body mesh
bpy.ops.wm.ply_import(filepath=body_path)
body = bpy.context.selected_objects[0]
body.name = "Body"

# Import the garment mesh
bpy.ops.wm.ply_import(filepath=garment_path)
garment = bpy.context.selected_objects[0]
garment.name = "Garment"

# Add cloth physics to the garment
bpy.context.view_layer.objects.active = garment
bpy.ops.object.modifier_add(type='CLOTH')
cloth_modifier = garment.modifiers["Cloth"]

# Set cloth simulation properties (you can tweak these)
cloth_modifier.settings.quality = 5  # Simulation quality
cloth_modifier.settings.mass = 0.5  # Mass of the cloth
cloth_modifier.settings.air_damping = 5  # Air resistance
cloth_modifier.settings.tension_stiffness = 10  # Stretch resistance
cloth_modifier.settings.compression_stiffness = 10  # Compression resistance
cloth_modifier.settings.shear_stiffness = 5  # Shear resistance
cloth_modifier.settings.bending_stiffness = 0.5  # Bending resistance

# Enable collisions for the body
bpy.context.view_layer.objects.active = body
bpy.ops.object.modifier_add(type='COLLISION')
collision_modifier = body.modifiers["Collision"]

# Run the simulation
bpy.context.view_layer.objects.active = garment
bpy.ops.screen.frame_jump(end=False)  # Go to the first frame
bpy.context.scene.frame_end = 50  # Duration of simulation
bpy.ops.screen.animation_play()

# Wait for the simulation to complete
bpy.ops.screen.animation_cancel()  # Stop playback

# Apply the simulation to the garment
bpy.ops.object.modifier_apply(modifier="Cloth")

# Export the simulated garment
bpy.ops.wm.ply_export(filepath=output_path)