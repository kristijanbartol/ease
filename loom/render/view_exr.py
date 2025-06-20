import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt

def load_exr(filename):
    """
    Load an EXR file and return it as a numpy array
    """
    file = OpenEXR.InputFile(filename)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B) = [np.frombuffer(file.channel(c, FLOAT), dtype=np.float32) for c in "RGB"]

    # Reshape into image format
    rgb = np.zeros((size[1], size[0], 3), dtype=np.float32)
    rgb[:,:,0] = np.reshape(R, (size[1], size[0]))
    rgb[:,:,1] = np.reshape(G, (size[1], size[0]))
    rgb[:,:,2] = np.reshape(B, (size[1], size[0]))

    return rgb

def display_exr(filename, exposure=0.0, gamma=2.2):
    """
    Display an EXR file with adjustable exposure and gamma
    """
    # Load the image
    rgb = load_exr(filename)
    
    # Apply exposure adjustment
    rgb *= 2**exposure
    
    # Apply gamma correction
    rgb = np.clip(rgb, 0, None)  # Clip negative values
    rgb = rgb**(1/gamma)
    
    # Clip to valid range
    rgb = np.clip(rgb, 0, 1)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb)
    plt.axis('off')
    plt.title(f'EXR Viewer (Exposure: {exposure}, Gamma: {gamma})')
    plt.show()

if __name__ == "__main__":
    # Example usage
    display_exr("render.exr", exposure=0.0, gamma=2.2)
    
    # You can adjust exposure and gamma for different views
    #display_exr("studio.exr", exposure=2.0, gamma=2.2)  # Brighter view
    #display_exr("studio.exr", exposure=-2.0, gamma=2.2)  # Darker view