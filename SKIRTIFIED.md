## Run "skirtified" mode

When the skirtified option is selected through the command line, i.e., design configuration, the message will be received that the skirtified body meshes first have to be created.
Below are the required steps.

### Run `main.py` with skirtified option

In the corresponding *.json file, the skirtified flag has to be true and the type has to be "dress" (for now, skirts are not yet implemented). As an example, see `dress.json` design.
Before the message is received, the original, SMPL body meshes with the specified body poses and shapes will be created in `data/skirtified/original/<design>-<pose_set>`.

### Generate skirtified meshes in Blender

If not already available, download and install Blender (recommended version is 4.1). Open the Blender project included in this repository at `src/blender/Skirtify/skirtify.blend`.
Install `torch` package for Blender to run the `skirtify.py` script within Blender (under "Scripting" tab). Once the script is successfully executed, the "Skirtify" command can be used
within "Geometry Nodes" tab by selecting the whole mesh and pressing F3 -> Skirtify. The skirtified mesh can then be exported in File->Export as PLY in the 
`data/skirtified/skirtified/<design>-<pose_set>/` directory.

This procedure has to be repeated for all the meshes (init and targets).

### Run `main.py` again

Once all the skirtified meshes are generated and exported, the `main.py` can finally be executed "normally" and the script should run all the steps (design, parameterization, 
visualization) and the PNG images should be stored.
