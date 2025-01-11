from tailorlang.eval.qualitative import qualitative_evaluation
from tailorlang.eval.quantitative import quantitative_evaluation
from tailorlang.render.simple_renderer import render_simple
from tailorlang.utils import construct_experiment_name
from tailorlang.eval.postprocess import postprocess_embedded
from tailorlang.blender.blender_caller import simulate_frame

from anisotropic_simulations.evaluation_frame_based import (
    get_non_skintight_garment,
    evaluate_nonskintight_jacobians
)


def evaluate_experiment(config):
    postprocess_embedded()
    experiment_name = construct_experiment_name(config)
    qualitative_evaluation(experiment_name)
    #quantitative_evaluation(experiment_name)
    sim_garment_dict = get_non_skintight_garment(project_dir=config.project_dir)
    evaluate_nonskintight_jacobians(sim_garment_dict)
    simulate_frame(project_dir=config.project_dir)
    #render_simple(experiment_name)  # TODO: ALso use qualitative stretch colors for colored rendering to see the stretch errors
