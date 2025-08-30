import os
import subprocess
from sys import platform


def run_parameterization(config):
    # Get absolute paths
    current_dir = os.getcwd()
    if platform == 'darwin':
        cpp_program_path = os.path.join(current_dir, "anisotropic-parameterization/cmake-build-debug/optimize_set_with_seamlines")
    else:
        cpp_program_path = os.path.join(current_dir, "anisotropic-parameterization/build/optimize_set_with_seamlines")
    root_project_path_arg = os.path.abspath(config.project_dir)
    
    # Ensure the executable exists
    if not os.path.exists(cpp_program_path):
        raise FileNotFoundError(f"Executable not found at: {cpp_program_path}")
    
    # Start with base command and config file
    command = [
        cpp_program_path,
        "--config", "anisotropic-parameterization/configs/default.json"
    ]
    
    # Add boolean flags if they are True
    if config.optim_dress:
        command.append("--optim-dress")
    if config.use_darts:
        command.append("--use-darts")
    if config.equalize_seamline_lengths:
        command.append("--equalize-lengths")
    
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
    
    for python_param, cpp_param in param_mapping.items():
        if hasattr(config, python_param):
            value = getattr(config, python_param)
            if value is not None:  # Only add if value is specified
                command.extend([cpp_param, str(value)])
    
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


def run_loom_optimization(hyperparams_config):
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

    for python_param, cpp_param in param_mapping.items():
        if python_param in hyperparams_config:
            value = hyperparams_config[python_param]
            if value is not None:  # Only add if value is specified
                command.extend([cpp_param, str(value)])
    
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
        

def run_remeshing(project_path):
    cpp_program_path = "./isotropic-remeshing/build/remesh"
    
    