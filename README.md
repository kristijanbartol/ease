# TailorLang

TailorLang is the root project based on the "Locally-Scaled Embedded Garment Meshes for Computational Fit" (currently under review). The preprint is uploaded to this repository (`assets/preprint_review.pdf`).

## Installation

### Download the necessary files

Go to the [SMPL website](https://smpl.is.tue.mpg.de/) -> Downloads (login required) and then download the file specified as
"Download version 1.0.0 for Python 2.7 (female/male. 10 shape PCs)". Unzip the contents in the selected location (this location will be used
as a command line argument for the `main.py`).

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

### "Skirtified" designs and bodies

To design dresses (and soon skirts), the skirtified body meshes should be separately prepared in Blender. The detailed procedure with images
is specified in the `SKIRTIFIED.md`.

## Known bugs and limitations

1. The lower garment patches are not properly cut to get smooth edges at the bottom of the pants. Instead, only some of the values in *.json files will be smooth.

2. Seamlines are not perfectly aligned and directly suitable for manufacturing.

3. There are parameter values that will result in errors (when the values are smaller than the minimal "possible" value to cut).
