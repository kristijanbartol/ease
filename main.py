import json
from sys import platform

from loom.mesh_processing import MeshState
from loom.eval.experiment import loom_evaluate_experiment
from loom.utils import (
    construct_configs,
    prepare_configuration,
    print_configuration
)

from loom_core.params import process_config
from loom_core.cut import run_design
from loom_core.optimize import run_loom_optimization
from loom_core.evaluation.experiment import evaluate_experiment


def run_loom_core():
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

    run_design(SMPL_DIR, design_params, body_set, is_dress, is_skirt)
    run_loom_optimization(hyperparams)
    evaluate_experiment(PROJECT_DIR, SMPL_DIR, experiment_name, design_params, body_set, is_dress, is_skirt)


def run_loom():
    init_config = prepare_configuration()
    print_configuration(init_config)
    
    grid_configs = construct_configs(init_config=init_config)    # TODO: Also add stretch & sliding parameters
     
    for config in grid_configs:     
        mesh_state = MeshState(config=config)
        #mesh_state.update_parameters(config.design)     # TODO: When measuring execution time, measure only the update part (init will be stored and loaded)
        mesh_state.finalize()
        mesh_state.optimize()
        
        loom_evaluate_experiment(config, mesh_state.design_params, mesh_state.body_set)


if __name__ == '__main__':
    #run_loom()
    run_loom_core()    
