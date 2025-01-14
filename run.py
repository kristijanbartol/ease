from tailorlang.mesh_processing import MeshState
from tailorlang.eval.eval import evaluate_experiment
from tailorlang.utils import (
    construct_configs,
    construct_experiment_name,
    prepare_configuration,
    print_configuration
)


def run_grid(init_config):
    grid_configs = construct_configs(init_config=init_config)    # TODO: Also add stretch & sliding parameters
     
    for config in grid_configs:     
        mesh_state = MeshState(config=config)
        #mesh_state.update_parameters(config.design)     # TODO: When measuring execution time, measure only the update part (init will be stored and loaded)
        mesh_state.finalize()
        mesh_state.optimize()
        
        evaluate_experiment(config)


if __name__ == "__main__":
    init_config = prepare_configuration()
    print_configuration(init_config)
    
    run_grid(init_config=init_config)
