from tailorlang.eval.qualitative import qualitative_evaluation
from tailorlang.eval.quantitative import quantitative_evaluation
from tailorlang.render.simple_renderer import render_simple
from tailorlang.utils import construct_experiment_name
from tailorlang.eval.postprocess import postprocess
from tailorlang.sim.runner import simulate_garment_set


def evaluate_experiment(config, design_params, body_set):
    experiment_name = construct_experiment_name(config)
    stacked_uv_coords_dict = postprocess(experiment_name)
    qualitative_evaluation(experiment_name)
    #quantitative_evaluation(experiment_name)
    simulate_garment_set(config, experiment_name, design_params, body_set, stacked_uv_coords_dict)
    #render_simple(experiment_name)  # TODO: ALso use qualitative stretch colors for colored rendering to see the stretch errors
