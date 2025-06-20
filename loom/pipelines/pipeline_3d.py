import json

from loom.sim.simulator_3d_pipeline import simulate_3d
from loom.utils import prepare_configuration
from loom.garment import (
    BodySet,
    DesignParameters
)


def run_pipeline():
    config = prepare_configuration()

    with open(f'config/designs/{config.design}.json', 'r') as json_file:
        init_design_dict = json.load(json_file)
    design_params = DesignParameters(init_design_dict)

    with open(f'config/body_sets/{config.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)
    body_set = BodySet(set_dict)
    
    simulate_3d(config, design_params, body_set)


if __name__ == '__main__':
    run_pipeline()
