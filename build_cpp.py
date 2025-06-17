# build_cpp.py
import os
import subprocess
import sys
from pathlib import Path

def build_cpp_project():
    # Get the directory containing this script
    root_dir = Path(__file__).parent.absolute()
    cpp_dir = root_dir / 'anisotropic-parameterization'
    build_dir = cpp_dir / 'build'
    
    # Create lib directory in the Python package
    lib_dir = root_dir / 'tailorlang' / 'lib'
    lib_dir.mkdir(exist_ok=True)
    print(f"Created lib directory at: {lib_dir}")  # Debug print

    # Create build directory if it doesn't exist
    build_dir.mkdir(exist_ok=True)
    print(f"Building in directory: {build_dir}")  # Debug print

    try:
        # Print current working directory for debugging
        print(f"Current working directory: {os.getcwd()}")
        
        # Run cmake with verbose output
        print("Running CMake...")
        result = subprocess.run(['cmake', '..'], 
                              cwd=str(build_dir), 
                              capture_output=True, 
                              text=True)
        print(f"CMake output:\n{result.stdout}\n{result.stderr}")
        
        # Run make with verbose output
        print("Running Make...")
        result = subprocess.run(['make'], 
                              cwd=str(build_dir), 
                              capture_output=True, 
                              text=True)
        print(f"Make output:\n{result.stdout}\n{result.stderr}")
        
        # Get the path to the compiled module
        if sys.platform == 'darwin':  # macOS
            module_name = 'remesh_module.so'
        elif sys.platform == 'win32':  # Windows
            module_name = 'remesh_module.pyd'
        else:  # Linux
            module_name = 'remesh_module.so'
            
        module_path = build_dir / module_name
        print(f"Looking for module at: {module_path}")  # Debug print
        
        if module_path.exists():
            print(f"Found compiled module at: {module_path}")
        else:
            print(f"Warning: Compiled module not found at expected location!")
            # List directory contents
            print("Build directory contents:")
            for item in build_dir.iterdir():
                print(f"  {item}")
        
        # Copy the compiled module to the Python package
        dest_path = lib_dir / module_name
        if module_path.exists():
            import shutil
            shutil.copy2(str(module_path), str(dest_path))
            print(f"Copied module to: {dest_path}")
            return str(dest_path)
        else:
            raise FileNotFoundError(f"Compiled module not found at {module_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error building C++ project: {e}")
        print(f"Command output:\n{e.output}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == '__main__':
    build_cpp_project()