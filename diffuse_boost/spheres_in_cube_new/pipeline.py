from diffuse_boost import cfg

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

if __name__ == "__main__":
    # Avoid Circular imports
    from diffuse_boost.spheres_in_cube_new.data_generation import _get_cfg
    from diffuse_boost.spheres_in_cube_new import data_generation
    from diffuse_boost.spheres_in_cube_new import flow_matching_spheres
    from diffuse_boost.spheres_in_cube_new import plot_data_spheres

    iterations = _get_cfg("spheres_in_cube_new_pipeline", "iterations", 1)
    start_at_step = _get_cfg("spheres_in_cube_new_pipeline", "start_at_step", "push")

    #"start", "train_and_sample", "push", "retrain_and_sample"
    state = PipelineState()
    print(f"[Pipeline] Start step: {start_at_step}")
    
    # Start outside the main loop
    if start_at_step == "start":
        cfg.set("sample_generation_PP+PBTS", "mode", "training_set_gen")
        data_generation.main(state=state)
    elif start_at_step == "push":
        print(f"[Pipeline] [Start Push]")
        cfg.set("sample_generation_PP+PBTS", "mode", "final_push")
        data_generation.main(state=state)
        cfg.set("flow_matching", "dataset_path", state.pushed_samples_path)

    for i in range(iterations):
        print(f"[Pipeline] Iteration ({i+1}/{iterations})")

        # (Re)train Model
        if i == 0 and start_at_step == "train_and_sampling":
            cfg.set("flow_matching", "mode", "train_and_sampling")
        else:
            cfg.set("flow_matching", "mode", "retrain_and_sampling")
        print(f"[Pipeline] [Start retraining and sampling] - Iteration ({i+1}/{iterations})")
        flow_matching_spheres.main(state=state)
        cfg.set("flow_matching", "resume_model_path", state.model_path)
        cfg.set("sample_generation_PP+PBTS", "final_push_input", state.samples_path)

        # Final push Samples
        print(f"[Pipeline] [Start Push] - Iteration ({i+1}/{iterations})")
        cfg.set("sample_generation_PP+PBTS", "mode", "final_push")
        data_generation.main(state=state)
        cfg.set("flow_matching", "dataset_path", state.pushed_samples_path)


            # 1000 Samples --> 5000 pushed --> top 1000