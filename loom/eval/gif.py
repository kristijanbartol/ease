import cv2
import imageio
import os
from pathlib import Path

from loom.utils import (
    get_experiment_names_for_grid,
    prepare_configuration,
    print_configuration
)


def create_gif_from_images(image_paths, output_path, fps=2):
    """
    Create a GIF from a list of image paths.
    
    Args:
        image_paths (list): List of full paths to PNG images
        output_path (str): Path where the GIF will be saved
        duration (float): Duration for each frame in seconds
    """
    # Read all images
    images = []
    for image_path in image_paths:
        # Read image using cv2
        img = cv2.imread(str(image_path))
        
        # Convert from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        images.append(img_rgb)
    
    # Create GIF
    imageio.mimsave(
        output_path,
        images,
        fps=2,
        loop=0  # 0 means loop forever
    )


def keywords_in_subdir(keywords, subdir):
    for keyword in keywords:
        if keyword not in subdir:
            return False
    return True


def name_from_keywords(keywords):
    file_name = keywords[0]
    for keyword in keywords:
        file_name += '_' + keyword
    return file_name


if __name__ == '__main__':
    png_rootdir = 'results/qualitative/pattern/png'
    
    init_config = prepare_configuration()
    print_configuration(init_config)
    
    experiment_names = get_experiment_names_for_grid(init_config=init_config)
    
    output_rootdir = 'results/qualitative/pattern/gif/'
    output_path = os.path.join(output_rootdir, f'{init_config.group_label}.gif')
    
    os.makedirs(output_rootdir, exist_ok=True)
    
    img_paths = []
    for experiment_subdir in experiment_names:
        experiment_dirpath = os.path.join(png_rootdir, experiment_subdir)
        img_path = os.path.join(experiment_dirpath, 'sewing_pattern_final-seams.png')
        img_paths.append(img_path)
    
    create_gif_from_images(img_paths, output_path)
