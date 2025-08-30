from sys import platform
import os
import subprocess


def run_loom_optimization():
    # Get absolute paths
    current_dir = os.getcwd()
    if platform == 'darwin':
        cpp_program_path = os.path.join(current_dir, "anisotropic-parameterization/cmake-build-debug/loom")
    else:
        cpp_program_path = os.path.join(current_dir, "anisotropic-parameterization/build/loom")
    root_project_path_arg = os.path.abspath(current_dir)
    
    # Ensure the executable exists
    if not os.path.exists(cpp_program_path):
        raise FileNotFoundError(f"Executable not found at: {cpp_program_path}")
    
    # Start with base command and config file
    command = [
        cpp_program_path,
        "--config", "anisotropic-parameterization/configs/default.json"
    ]
    
    # Add other parameters with their values
    param_mapping = {
        "matching_mode": "--matching-mode",
        "seamline_strategy": "--seamline-strategy",
        "num_seam_iters": "--num-seam-iters",
        "max_stretch": "--max-stretch",
        "material_stretch_coef": "--material-stretch-coef",
        "stretch_coef": "--stretch-coef",
        "edges_coef": "--edges-coef",
        "seams_coef": "--seams-coef",
        "dart_coef": "--dart-coef"
    }
    
    print(f"Running command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True,
            env=os.environ.copy()
        )
        print(f"Program output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the C++ program: {e}")
        print(f"Program output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
