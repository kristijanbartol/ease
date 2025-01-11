import torch
import trimesh
import os
import numpy as np
from copy import deepcopy
from smplx import SMPL

from tailorlang.const import SMPL_DIR
from tailorlang.anthropometry.measure import MeasureSMPL


SEARCH_GRID = [
    torch.linspace(0.79, 0.81, 20), #[0.7980],
    torch.linspace(0.865, 0.88, 15), #[0.8737],
    torch.linspace(-0.69, -0.71, 20),    # -0.7020
    torch.linspace(0.0, 2.0, 100),
    torch.linspace(-1.0, 1.0, 100),
    torch.linspace(0.0, 0.0, 1),
    [0.0],
    [0.0],
    [0.0],
    [0.0],
]

# [ 0.8100,  0.8800, -0.7100,  0.5000, -0.1667,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000]
# [ 0.8100,  0.8800, -0.7100,  1.0000,  0.1000,  0.2500,  0.0000,  0.0000, 0.0000,  0.0000]
# [ 0.8100,  0.8800, -0.7100,  0.2424,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000]


def get_smpl_from_measures(project_dir, subject_measure_dict):
    #smpl_model = SMPL(
    #    model_path=os.path.join(SMPL_DIR, f'SMPL_MALE.pkl'), 
    #    gender='male'
    #)
    measurer = MeasureSMPL(project_dir=project_dir, smpl_dir=SMPL_DIR)
    latest_betas = torch.zeros((1, 10))
    min_error = 1000
    best_errors = None
    best_idxs = np.zeros(10, dtype=np.int16)
    best_measurements = None
    best_betas = None
    for beta_idx in range(10):
        for item_idx, beta in enumerate(SEARCH_GRID[beta_idx]):
            latest_betas[0, beta_idx] = beta
            measurer.from_body_model(gender='male', shape=latest_betas)
            measurement_names = measurer.all_possible_measurements
            measurer.measure(measurement_names)
            
            relevant_pairs = []
            for measure_name in subject_measure_dict:
                relevant_pairs.append([subject_measure_dict[measure_name], measurer.measurements[measure_name]])
                print(f'{measure_name}: {abs(subject_measure_dict[measure_name] - measurer.measurements[measure_name])}')
            relevant_pairs = np.array(relevant_pairs)
            mean_error = np.mean(np.abs(relevant_pairs[:, 0] - relevant_pairs[:, 1]))
            if mean_error < min_error and abs(subject_measure_dict['height'] - measurer.measurements['height']) < 0.3 and abs(subject_measure_dict['weight'] - measurer.measurements['weight']) < 0.4:
                min_error = mean_error
                best_idxs[beta_idx] = item_idx
                best_errors = relevant_pairs[:, 0] - relevant_pairs[:, 1]
                best_measurements = deepcopy(measurer.measurements)
                best_betas = deepcopy(latest_betas)
            print(f'beta sample #{item_idx} (PC{beta_idx})')
            
    print(min_error)
    print(best_measurements)
    print(best_errors)
    print(best_betas)
    model = SMPL(
        model_path=os.path.join(SMPL_DIR, f'SMPL_MALE.pkl'), 
        gender='male'
    )
    smpl_output = model(betas=best_betas)
    trimesh.Trimesh(vertices=smpl_output.vertices.detach().cpu().numpy().squeeze(), faces=model.faces).export('subject_mesh.ply')


if __name__ == '__main__':
    subject_measure_dict = {
        'height': 172,
        'weight': 65.5,
        'waist circumference': 77,
        'hip circumference': 93,
        #'chest circumference': 89.5,
        'bicep circumference': 28.5,
        'forearm circumference': 26.5,
        'thigh circumference': 46,
        'calf circumference': 37
    }
    get_smpl_from_measures('/Users/kristijanbartol/Tailorlang/', subject_measure_dict)
