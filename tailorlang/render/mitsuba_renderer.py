import mitsuba as mi
import drjit as dr
import numpy as np
from mitsuba.scalar_rgb import Transform4f, load_dict
from tailorlang.render.materials import (
    create_asian_skin_bsdf,
    create_caucasian_skin_bsdf,
    create_cotton_bsdf,
    create_dark_skin_bsdf,
    create_denim_bsdf,
    create_silk_bsdf
)

def setup_scene(body_path, garment_path, output_path):
    """
    Set up the Mitsuba scene with body and garment meshes
    """
    # Initialize Mitsuba
    mi.set_variant('scalar_rgb')
    
    # Create scene dictionary
    scene_dict = {
        'type': 'scene',
        
        # Add camera
        'camera': {
            'type': 'perspective',
            'fov': 45,
            'to_world': Transform4f.look_at(
                origin=[0, 0, 5],    # Camera position
                target=[0, 0, 0],    # Looking at center
                up=[0, 1, 0]         # Up vector
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 1920,
                'height': 1080,
                'pixel_format': 'rgb',
                'rfilter': {'type': 'gaussian'}
            },
        },
        
        # Add lighting
        'light': {
            'type': 'envmap',
            'filename': 'studio.exr',  # Replace with your environment map
            'scale': 2.0
        },
        
        # Add body mesh
        'body': {
            'type': 'ply',
            'filename': body_path,
            'bsdf': create_dark_skin_bsdf()
        },
        
        # Add garment mesh
        'garment': {
            'type': 'ply',
            'filename': garment_path,
            'bsdf': create_cotton_bsdf()
        },
        
        # Add integrator
        'integrator': {
            'type': 'path',
            'max_depth': 8
        }
    }
    
    return scene_dict

def render_scene(scene_dict, output_path, spp=128):
    """
    Render the scene and save the output
    """
    # Load the scene
    scene = load_dict(scene_dict)
    
    # Render
    image = mi.render(scene, spp=spp)
    
    # Save the rendered image
    mi.util.write_bitmap(output_path, image)
    return image

def main():
    # Define paths
    body_path = 'data/body/ref.ply'
    garment_path = 'upper_deformed.ply'
    output_path = 'render.exr'
    
    # Set up scene
    scene_dict = setup_scene(body_path, garment_path, output_path)
    
    # Render
    render_scene(scene_dict, output_path, spp=256)

if __name__ == '__main__':
    main()