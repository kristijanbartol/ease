import os
import subprocess


def run_parameterization(project_path, use_darts):
    # Get absolute paths
    current_dir = os.getcwd()
    cpp_program_path = os.path.join(current_dir, "anisotropic-parameterization/build/optimize_set_with_seamlines")
    root_project_path_arg = os.path.abspath(project_path)
    
    # Ensure the executable exists
    if not os.path.exists(cpp_program_path):
        raise FileNotFoundError(f"Executable not found at: {cpp_program_path}")
    
    # Verify required directories exist
    required_dirs = [
        os.path.join(root_project_path_arg, "data/embedded/ui/"),
        os.path.join(root_project_path_arg, "data/seamlines/ui"),
        os.path.join(root_project_path_arg, "data/darts/latest")
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Required directory not found: {dir_path}")
    
    optim_dress_arg = "0"
    use_darts_arg = "1" if use_darts else "0"
    
    command = [cpp_program_path, root_project_path_arg, optim_dress_arg, use_darts_arg]
    print(f"Running command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True,
            env=os.environ.copy()  # Ensure environment variables are passed
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
    
    