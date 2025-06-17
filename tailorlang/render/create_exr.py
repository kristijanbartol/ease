import numpy as np
from PIL import Image
import OpenEXR
import Imath
import array

def create_studio_environment():
    """
    Create a studio-like environment map with key, fill, and rim lights
    """
    # Create HDR canvas (height x width x RGB)
    width = 2048
    height = 1024
    env_map = np.zeros((height, width, 3), dtype=np.float32)
    
    # Helper function to add a light source
    def add_light(x, y, intensity, size, color):
        y_coords, x_coords = np.ogrid[-y:height-y, -x:width-x]
        mask = x_coords*x_coords + y_coords*y_coords <= size*size
        env_map[mask] += np.array(color) * intensity
    
    # Add key light (main light)
    add_light(
        x=width//4,
        y=height//3,
        intensity=5.0,
        size=100,
        color=[1.0, 0.98, 0.95]
    )
    
    # Add fill light (softer, secondary light)
    add_light(
        x=3*width//4,
        y=height//3,
        intensity=2.0,
        size=150,
        color=[0.9, 0.9, 1.0]
    )
    
    # Add rim light (background highlight)
    add_light(
        x=width//2,
        y=2*height//3,
        intensity=3.0,
        size=80,
        color=[1.0, 1.0, 1.0]
    )
    
    # Add ambient light
    env_map += 0.1
    
    return env_map

def save_exr(filename, env_map):
    """
    Save the environment map as an EXR file
    """
    # Prepare header
    header = OpenEXR.Header(env_map.shape[1], env_map.shape[0])
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, half_chan) for c in "RGB"])
    
    # Create output file
    out = OpenEXR.OutputFile(filename, header)
    
    # Write pixel data
    r = array.array('f', env_map[:,:,0].flatten()).tobytes()
    g = array.array('f', env_map[:,:,1].flatten()).tobytes()
    b = array.array('f', env_map[:,:,2].flatten()).tobytes()
    
    out.writePixels({'R': r, 'G': g, 'B': b})
    out.close()

def main():
    # Create environment map
    env_map = create_studio_environment()
    
    # Save as EXR
    save_exr("studio.exr", env_map)
    
    # Optional: Save a preview (tone-mapped version)
    preview = np.clip(env_map ** (1/2.2), 0, 1)
    Image.fromarray((preview * 255).astype(np.uint8)).save("preview.png")

if __name__ == "__main__":
    main()