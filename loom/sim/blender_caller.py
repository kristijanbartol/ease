import subprocess
import os

def simulate_pose(body_path, shirt_path, pant_path, 
                  body_output, shirt_output, pant_output,
                  blender_path, scripts_dir):
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
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("Simulation completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running simulation:")
        print(e.stderr)
        raise

if __name__ == "__main__":
    simulate_pose(
        body_path="/Users/kristijanbartol/TailorLang/data/body/target-00_x10.ply",
        shirt_path="/Users/kristijanbartol/TailorLang/results/non-skintight/sit-pose_long_bezier_1_2.0_1.0_2.0FFF/base_upper.ply",
        pant_path="/Users/kristijanbartol/TailorLang/results/non-skintight/sit-pose_long_bezier_1_2.0_1.0_2.0FFF/base_lower.ply",
        body_output="/Users/kristijanbartol/TailorLang/results/sim/body.ply",
        shirt_output="/Users/kristijanbartol/TailorLang/results/sim/shirt.ply",
        pant_output="/Users/kristijanbartol/TailorLang/results/sim/pant.ply",
        blender_path="/Applications/Blender.app/Contents/MacOS/Blender",
        scripts_dir='/Users/kristijanbartol/TailorLang/tailorlang/blender/'
    )
