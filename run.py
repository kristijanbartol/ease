from tailorlang.mesh_processing import MeshState
from tailorlang.eval.eval import evaluate_experiment
from tailorlang.utils import (
    construct_configs,
    construct_experiment_name,
    prepare_configuration,
    print_configuration
)


def run_grid(config):
    grid_configs = construct_configs(init_config=config)    # TODO: Also add stretch & sliding parameters
     
    for config in grid_configs:     
        mesh_state = MeshState(config=config)
        #mesh_state.update_parameters()     # TODO: When measuring execution time, measure only the update part (init will be stored and loaded)
        mesh_state.finalize()
        mesh_state.optimize()
        
        evaluate_experiment(config)


if __name__ == "__main__":
    config = prepare_configuration()
    print_configuration(config)
    
    run_grid(config=config)
