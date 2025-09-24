import subprocess
import os
from sys import platform

def simulate_pose(body_path, shirt_path, pant_path, 
                  body_output, shirt_output, pant_output,
                  is_dress, is_shoulderless, scripts_dir):
    """
    Run the cloth simulation in Blender
    
    Args:
        body_path (str): Path to body mesh
        shirt_path (str): Path to shirt mesh
        pant_path (str): Path to pant mesh
        body_output (str): Output path for body mesh
        shirt_output (str): Output path for shirt mesh
        pant_output (str): Output path for pant mesh
        blender_path (str): Path to Blender executable
    """
    if platform == 'darwin':
        blender_path = '/Applications/Blender.app/Contents/MacOS/Blender'
    else:
        blender_path = '/snap/bin/blender'

    # Construct the command
    cmd = [
        blender_path,
        "--background",  # Run Blender in background mode
        "--python", os.path.join(scripts_dir, 'blender_sim.py'),
        "--",  # Separator for script arguments
        "--body", body_path,
        "--shirt", shirt_path,
        "--pant", pant_path,
        "--body-output", body_output,
        "--shirt-output", shirt_output,
        "--pant-output", pant_output
    ]

    if is_dress:
        cmd.append('--is-dress')

    if is_shoulderless:
        cmd.append('--shoulderless')
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            #stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE,
            text=True
        )
        print("Simulation completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running simulation:")
        print(e.stderr)
        raise
