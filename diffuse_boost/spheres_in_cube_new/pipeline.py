import diffuse_boost
import numpy
import os
from diffuse_boost import cfg
from rich.console import Console

"""
Pipeline that loops Training --> Sampling --> Pushing --> Training --> ...

####  Important parameters to check before starting the pipeline  ####

[sample_generation_PP+PBTS]
num_spheres
sphere_radius
num_samples             (if start_at_step == start)
num_srp_restarts
final_push_input        (if start_at_step == push)

[flow_matching]
num_spheres
sphere_radius
num_epochs
learning_rate
dataset_path            (if start_at_step == train_and_sampling or start_at_step == retrain_and_sampling)
resume_model_path       (if start_at_step == retrain_and_sampling)
num_generated_samples
train_top_fraction
wall_weight

########################################################################
"""

class PipelineState:
    model_path:str = ""
    samples_path:str = ""
    pushed_samples_path:str = ""

    def set_model_path(self, model_path):
        self.model_path = model_path

    def set_samples_path(self, samples_path):
        self.samples_path = samples_path

    def set_pushed_samples_path(self, pushed_samples_path):
        self.pushed_samples_path = pushed_samples_path

if __name__ == "__main__":
    # Avoid Circular imports
    from diffuse_boost.spheres_in_cube_new.data_generation import _get_cfg, _set_cfg
    from diffuse_boost.spheres_in_cube_new import data_generation
    from diffuse_boost.spheres_in_cube_new import flow_matching_spheres
    from diffuse_boost.spheres_in_cube_new import plot_data_spheres

    # Color 
    console = Console()

    iterations = _get_cfg("spheres_in_cube_new_pipeline", "iterations", 1)
    start_at_step = _get_cfg("spheres_in_cube_new_pipeline", "start_at_step", "push")

    #"start", "train_and_sample", "push", "retrain_and_sample"
    state = PipelineState()
    console.print(f"[Pipeline] Start step: {start_at_step}", style="blue")
    
    # Start outside the main loop
    if start_at_step == "start":
        _set_cfg("sample_generation_PP+PBTS", "mode", "training_set_gen")
        data_generation.main(state=state)
        # Ensure flow-matching trains on the dataset we just generated.
        if not state.samples_path:
            raise RuntimeError("[Pipeline] training_set_gen completed but state.samples_path was not set.")
        if not os.path.exists(state.samples_path):
            raise FileNotFoundError(f"[Pipeline] Generated training dataset not found on disk: '{state.samples_path}'")
        _set_cfg("flow_matching", "dataset_path", state.samples_path)
    elif start_at_step == "push":
        console.print(f"[Pipeline] [Start Push]", style="blue")
        _set_cfg("sample_generation_PP+PBTS", "mode", "final_push")
        data_generation.main(state=state)
        _set_cfg("flow_matching", "dataset_path", state.pushed_samples_path)

    for i in range(iterations):
        console.print(f"[Pipeline] Iteration ({i+1}/{iterations})", style="blue")

        # (Re)train Model
        # If we started from scratch (start) or explicitly requested training, do a clean train on the first loop.
        if i == 0 and start_at_step in ("start", "train_and_sampling"):
            _set_cfg("flow_matching", "mode", "train_and_sampling")
        else:
            _set_cfg("flow_matching", "mode", "retrain_and_sampling")

        console.print(f"[Pipeline] [Start retraining and sampling] - Iteration ({i+1}/{iterations})", style="blue")
        flow_matching_spheres.main(state=state)
        _set_cfg("sample_generation_PP+PBTS", "final_push_input", state.samples_path)
        _set_cfg("flow_matching", "resume_model_path", state.model_path)

        # Final push Samples
        console.print(f"[Pipeline] [Start Push] - Iteration ({i+1}/{iterations})", style="blue")
        _set_cfg("sample_generation_PP+PBTS", "mode", "final_push")
        data_generation.main(state=state)
        _set_cfg("flow_matching", "dataset_path", state.pushed_samples_path)


            # plot_data_samples no input
            # plot_data_samples speed up

            # pipeline cases check
            # remove bloat .csv metrics ect
            # 
            # generate log file? (cant trust the console)
            # new data selection? right now 100% from new samples right?