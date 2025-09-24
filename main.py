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

    GENDER = 'female'

    with open('config/setup/loom.json') as config_f:
        config = json.load(config_f)
    experiment_name, design_params, hyperparams, body_set = process_config(config)

    NAME = "sit_average_10.0_50"

    run_design(SMPL_DIR, design_params, body_set)
    run_loom_optimization(hyperparams)
    evaluate_experiment(PROJECT_DIR, SMPL_DIR, experiment_name, design_params, body_set, optim_dress=False)
