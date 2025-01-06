from tailorlang.eval.qualitative import qualitative_evaluation
from tailorlang.eval.quantitative import quantitative_evaluation
from tailorlang.render.simple_renderer import render_simple
from tailorlang.utils import construct_experiment_name

from anisotropic_simulations.evaluation_frame_based import (
    get_single_frame_simulation_data,
    evaluate_simulation_frame
)


def evaluate_experiment(config):
    experiment_name = construct_experiment_name(config)
    qualitative_evaluation(experiment_name)
    quantitative_evaluation(experiment_name)
    sim_garment_dict = get_single_frame_simulation_data(project_dir=config.project_dir)
    evaluate_simulation_frame(sim_garment_dict)
    render_simple(experiment_name)  # TODO: ALso use qualitative stretch colors for colored rendering to see the stretch errors
