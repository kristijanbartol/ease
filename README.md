# TailorLang

TailorLang is the first DSL for parametric clothing with accurate cloth fitting. The project is motivated by the fact there is still no way to accurately determine how a piece of fabric fits the 3D body.
The central result for solving the fitting problem is to find "characteristic" pieces of fabric that fit specified body areas. As part of the TailorLang, the algorithm for finding the characteristic pieces
of fabric will be implemented.

🚧 This is an ongoing project. 🚧

## Installation

### Download the necessary files



### Conda environment

Create and activate conda environment:

```
conda create -n tailorlang-env python=3.8
conda activate tailorlang-env
```

Install requirements:

```
pip install -r requirements.txt
```

### Install dependency projects

Install garment-parameterization project:

```
git submodule update --init --recursive
```

One minor manual change is required in the libigl library. More specifically, the import `#include <cstdint>` has to be added after the `include<Eigen/Core>` and before `namespace igl` in the `garment-parameterization/lib/libigl/include/igl/opengl/MeshGL.h` file.

After this change, the `garment-parameterization` project can be built as:

```
cd garment-parameterization
mkdir build
cd build
cmake ..
make -j4 optimize_set_with_seamlines
```

## Running

Within the container, run `main.py` as:

```
python main.py --smpl_dir <path_to_smpl_dir> --body_set solo-female --design skintight --standard_export
```

To see all available command line arguments, use:

```
python main.py -h
```

or look directly at the `main.py` script. The `main.py` script generates embedded garment meshes, runs the parameterization algorithm implemented in C++ (`garment-parameterization` project), and visualizes the resulting, optimized sewing pattern. For more detailed description of the data flow, see `DATA.md`. 

### VSCode

Instead of running directly through the command line, it is recommended to use the Visual Studio Code (VSCode) IDE. The `./vscode/launch.json` file is already provided in this repository.

## Design parameter control

To produce different garment designs, the `*.json` files in `config/` directory should be used. There are two subdirectories. 

### Body sets

The `config/body_sets` subdirectory contains various definitions of body pose and shape sets. For simplicity, `solo-female.json` can always be used, specifying a single, average female body shape in A-pose.

### Designs

the `config/designs/` subdirectory contains various definitions of parametric designs. For starters, `skintight.json` can be used and edited. 
