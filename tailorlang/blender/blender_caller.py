import subprocess
import json
import os
import logging
from pathlib import Path
from typing import Dict, Optional

class BlenderClothSimulator:
    def __init__(self, blender_path: str, scripts_dir: Optional[str] = None):
        """
        Initialize the Blender cloth simulator.
        
        Args:
            blender_path: Path to Blender executable
            scripts_dir: Directory containing Blender scripts (defaults to current directory)
        """
        self.blender_path = Path(blender_path)
        self.scripts_dir = Path(scripts_dir) if scripts_dir else Path.cwd()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('BlenderClothSimulator')
        
        # Validate paths
        self._validate_setup()
    
    def _validate_setup(self) -> None:
        """Validate Blender executable and script paths."""
        if not self.blender_path.exists():
            raise FileNotFoundError(f"Blender executable not found at {self.blender_path}")
        
        blender_script = self.scripts_dir / 'sim.py'
        if not blender_script.exists():
            raise FileNotFoundError(f"Blender script not found at {blender_script}")
    
    def _validate_input_files(self, body_path: str, garment_path: str) -> None:
        """Validate input mesh files exist."""
        if not Path(body_path).exists():
            raise FileNotFoundError(f"Body mesh not found at {body_path}")
        if not Path(garment_path).exists():
            raise FileNotFoundError(f"Garment mesh not found at {garment_path}")
    
    def _create_output_directory(self, output_path: str) -> None:
        """Create output directory if it doesn't exist."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_simulation(
        self,
        body_path: str,
        garment_path: str,
        output_path: str,
        material_props: Optional[Dict] = None,
        frames: int = 250
    ) -> bool:
        """
        Run cloth simulation using Blender.
        
        Args:
            body_path: Path to body mesh OBJ file
            garment_path: Path to garment mesh OBJ file
            output_path: Path where the result should be saved
            material_props: Dictionary of material properties for the cloth
            frames: Number of simulation frames
        
        Returns:
            bool: True if simulation completed successfully
        """
        try:
            # Validate inputs
            self._validate_input_files(body_path, garment_path)
            self._create_output_directory(output_path)
            
            # Set default material properties if none provided
            if material_props is None:
                material_props = {
                    'mass': 0.3,
                    'tension': 15,
                    'compression': 15,
                    'shear': 5,
                    'bending': 0.5,
                    'friction': 0.5,
                    'tension_damping': 5,
                    'compression_damping': 5
                }
            
            # Prepare simulation parameters
            params = {
                'body_path': str(Path(body_path).absolute()),
                'garment_path': str(Path(garment_path).absolute()),
                'output_path': str(Path(output_path).absolute()),
                'material_props': material_props,
                'frames': frames
            }
            
            # Create temporary parameter file
            params_file = self.scripts_dir / 'sim_params.json'
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=2)
            
            self.logger.info(f"Starting cloth simulation with parameters: {params}")
            
            # Run Blender script
            result = subprocess.run([
                str(self.blender_path),
                '--background',
                '--python',
                str(self.scripts_dir / 'sim.py'),
                '--',
                str(params_file)
            ], capture_output=True, text=True)
            
            # Check for errors
            if result.returncode != 0:
                self.logger.error(f"Blender process failed: {result.stderr}")
                return False
            
            self.logger.info("Simulation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during simulation: {str(e)}")
            raise
        
        finally:
            # Clean up parameter file
            if 'params_file' in locals() and params_file.exists():
                params_file.unlink()

def simulate_frame(project_dir):
    # Example usage
    try:
        # Initialize simulator
        blender_path = r"/Applications/Blender.app/Contents/MacOS/Blender"
        scripts_dir = os.path.join(project_dir, 'tailorlang/blender/')
        simulator = BlenderClothSimulator(blender_path, scripts_dir)
        
        # Define paths
        body_path = r"data/body/ref.ply"
        garment_path = r"upper_deformed.ply"
        #output_path = r"simulated_garment.ply"
        output_path = "/Users/kristijanbartol/TailorLang/simulated_garment.ply"
        
        # Define material properties (cotton-like fabric)
        material_props = {
            'mass': 0.25,
            'tension': 20,
            'compression': 20,
            'shear': 10,
            'bending': 0.5,
            'friction': 0.45,
            'tension_damping': 5,
            'compression_damping': 5
        }
        
        # Run simulation
        success = simulator.run_simulation(
            body_path=body_path,
            garment_path=garment_path,
            output_path=output_path,
            material_props=material_props,
            frames=250
        )
        
        if success:
            print("Simulation completed successfully")
        else:
            print("Simulation failed")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    simulate_frame()
