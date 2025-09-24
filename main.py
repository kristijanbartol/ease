import json
from sys import platform

from loom_core.params import process_config
from loom_core.cut import run_design
from loom_core.optimize import run_loom_optimization
from loom_core.evaluation.experiment import evaluate_experiment


if __name__ == '__main__':
    if platform == 'darwin':
        PROJECT_DIR = '/Users/kristijanbartol/LOOM/'
        SMPL_DIR = '/Users/kristijanbartol/data/smpl/models/'
    else:
        PROJECT_DIR = '/home/kristijan/LOOM/'
        SMPL_DIR = '/home/kristijan/data/smpl/models/'

    with open('config/setup/loom.json') as config_f:
        config = json.load(config_f)
    experiment_name, design_params, hyperparams, body_set = process_config(config)

    is_dress = design_params['upper']['flag']['is_dress']
    is_skirt = design_params['lower']['flag']['is_skirt']
    is_skirtified = is_dress or is_skirt

    run_design(SMPL_DIR, design_params, body_set, is_skirtified)
    run_loom_optimization(hyperparams)
    evaluate_experiment(PROJECT_DIR, SMPL_DIR, experiment_name, design_params, body_set, optim_dress=is_skirtified)
