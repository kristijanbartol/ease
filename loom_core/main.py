import json
from sys import platform

from params import process_config
from cut import run_design
from optimize import run_loom_optimization
from evaluation.experiment import evaluate_experiment


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

    run_design(config)
    run_loom_optimization()
    evaluate_experiment(experiment_name, config, design_params, body_set)
