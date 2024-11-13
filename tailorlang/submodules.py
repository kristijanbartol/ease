import os
import subprocess


def run_parameterization(project_path, use_darts):
    cpp_program_path = "./anisotropic-parameterization/build/optimize_set_with_seamlines"

    root_project_path_arg = os.path.abspath(project_path)
    use_darts_arg = "1" if use_darts else "0"

    command = [cpp_program_path, root_project_path_arg, use_darts_arg]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the C++ program: {e}")
        print(f"Error output: {e.stderr}")
    except FileNotFoundError:
        print(f"The C++ program was not found at {cpp_program_path}")
        

def run_remeshing(project_path):
    cpp_program_path = "./isotropic-remeshing/build/remesh"
    
    