# NOTE: Currently deprecated, since I will not use Blender for cloth simulations.

'''
This Blender script generates individual shape parameters.

The script should be run within Blender and used along with SMPL-X plugin as follows:
1. Open the SMPL-X dialogue.
2. Select gender (recommended 50 females and 50 males in total).
3. Click Shape->Random to generate random shape.
4. Run the script to save the shape parameters of the randomly generated SMPL mesh.

The generated parameters will be used as reference shapes.
'''

import bpy
import numpy as np
import json
import os

DATA_DIR = './TailorLang/data/shapes/'
GENDER = 'female'  # NOTE: This is manual. It has to match the selected gender.


# Assume the SMPL-X model is the active object
obj = bpy.context.active_object

shape_array = np.zeros(10,)
for key in obj.data.shape_keys.key_blocks:
    if key.name.startswith('Shape'):
        beta_idx = int(key.name[-3:])
        shape_array[beta_idx] = key.value

shape_dict = { 'gender': GENDER }
for pc_idx in range(10):
    shape_dict['pc{}'.format(pc_idx)] = shape_array[pc_idx]
            
params_files = os.listdir(DATA_DIR)
    
file_indices = set()
for fname in params_files:
    if fname.startswith('params') and (fname.endswith('.json') or fname.endswith('.npy')):
        file_index = int(fname.split('.')[0][-3:])
        file_indices.add(file_index)

next_index = 0
while next_index in file_indices:
    next_index += 1
    
if next_index <= 100:
    index_str = '{:03d}'.format(next_index)

    json_fname = 'params{}.json'.format(index_str)
    npy_fname = 'params{}.npy'.format(index_str)

    with open(os.path.join(DATA_DIR, json_fname), 'w') as json_file:
        json.dump(shape_dict, json_file, indent=4, sort_keys=True)

    np.save(os.path.join(DATA_DIR, npy_fname), shape_array)

    print("Created files: {}, {}".format(json_fname, npy_fname))
