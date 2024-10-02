## Data organization details

There is a significant number of files generated during the execution of the `main.py` script. Different groups of files and the corresponding directories can be distinguished as detailed below.

## Garment patches and local stretches

### Generated directories

When `main.py` is executed, `data/embedded/latest` directory is automatically deleted so that the new "latest" directory can be created instead. The directory has two subdirectories (`skintight`
and `loose`) but only `skintight` is relevant for the majority of cases. Within `data/embedded/latest/skintight`, there is a set of subdirectories for each garment patch, namely:

- `lower_back_left`, `lower_back_right`, `lower_front_left`, `lower_front_right`, `upper_front`, `upper_back`, `sleeve_back_left`, `sleeve_back_right`, `sleeve_front_left`, `sleeve_front_right`
in case of default mode (non-skirtified);

- `upper_front` and `upper_back` in case of skirtified case.

In both cases, the `body` directory is created as well.

Along with the `data/embedded/latest/` directory, the directory named `<design>-<body_set>` (replaced by the corresponding selected design and body set) is also created and is identical to `latest/`.

### Embedded garment meshes files

In the `data/embedded/latest/skintight/<garment_patch>` directory, there are file names `init.ply` (and optionally `target-0.ply`, `target-1.ply` if multiple targets are specified). These 
PLY files contain the corresponding garment mesh patches (embedded on the skin, i.e., cut out from the SMPL mesh).

### Local triangle stretches files

Along with the PLY mesh files in the `data/embedded/latest/skintight/<garment_patch>` directory, two *.txt files are generated specifying local triangle stretches for each triangle on the 
garment  patch and passed to the parameterization algorithm (`stretches_u.txt` and `stretches_v.txt`). The other *.txt files are not relevant in most of the cases.

### Optimized 2D patch mesh files

Using the embedded garment meshes and the local triangle stretches, the parameterization algorithm generates `data/embedded/latest/skintight/<garment_patch>/optim*.ply` files. These meshes 
represent optimized 2D patches and will be used for specifying cloth simulation constraints. Currently, they are only visualized as 2D images in the next step (below).

### Images of optimized 2D patches

The final product of the `main.py` are the PNG images of the sewing pattern, stored in `data/embedded/latest/skintight/sewing_pattern_*.png`. Different PNG files represent the results after
different parameterization stages within the algorithm but only the final result is relevant.

## Seamlines

The seamline definitions are stored in `data/seamlines` directory. Same as with the `data/embedded` directory, the `latest/` directory contains the latest results. For each seamline, there
is a corresponding *.txt file with garment mesh indices defining the seamline.
