import json

from tailorlang.eval.eval import evaluate_experiment
from tailorlang.utils import prepare_configuration
from tailorlang.garment import (
    BodySet,
    DesignParameters
)


if __name__ == '__main__':
    config = prepare_configuration()

    with open(f'config/designs/{config.design}.json', 'r') as json_file:
        init_design_dict = json.load(json_file)
    design_params = DesignParameters(init_design_dict)

    with open(f'config/body_sets/{config.body_set}.json', 'r') as json_file:
        set_dict = json.load(json_file)
    body_set = BodySet(set_dict)
    
    evaluate_experiment(config, design_params, body_set)
