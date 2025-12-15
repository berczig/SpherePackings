import diffuse_boost
from diffuse_boost import cfg
from rich.console import Console

class PipelineState:
    model_path: str = ""
    samples_path: str = ""
    pushed_samples_path: str = ""

    def set_model_path(self, model_path):
        self.model_path = model_path

    def set_samples_path(self, samples_path):
        self.samples_path = samples_path

    def set_pushed_samples_path(self, pushed_samples_path):
        self.pushed_samples_path = pushed_samples_path


if __name__ == "__main__":
    # Avoid circular imports; matches your pipeline style.
    from diffuse_boost.circles_in_square.sample_generation import _get_cfg, _set_cfg
    from diffuse_boost.circles_in_square import sample_generation
    from diffuse_boost.circles_in_square import flow_matching_sumradii_centers_only

    console = Console()

    iterations = _get_cfg("circle_packing_pipeline", "iterations", 1)
    start_at_step = _get_cfg("circle_packing_pipeline", "start_at_step", "push")

    state = PipelineState()
    console.print(f"[Pipeline] Start step: {start_at_step}", style="blue")

    # ------------------------------------------------------------
    # Start outside main loop
    # ------------------------------------------------------------
    if start_at_step == "start":
        console.print("[Pipeline] [Start training set generation]", style="blue")

        _set_cfg("circle_packing_SRP", "mode", "training_set_gen")
        sample_generation.main(state=state)

        if getattr(state, "samples_path", ""):
            _set_cfg("flow_matching_sumradii_centers_only", "dataset_path", state.samples_path)
        else:
            raise RuntimeError("SRP generation finished, but state.samples_path was not set.")

    elif start_at_step == "push":
        console.print("[Pipeline] [Start Push]", style="blue")

        _set_cfg("circle_packing_SRP", "mode", "push_only")
        sample_generation.main(state=state)

        if getattr(state, "pushed_samples_path", ""):
            _set_cfg("flow_matching_sumradii_centers_only", "dataset_path", state.pushed_samples_path)
        else:
            raise RuntimeError("SRP push finished, but state.pushed_samples_path was not set.")

    # ------------------------------------------------------------
    # Main loop: Train+Sample -> Push -> repeat
    # ------------------------------------------------------------
    for i in range(iterations):
        console.print(f"[Pipeline] Iteration ({i+1}/{iterations})", style="blue")

        if i == 0:
            _set_cfg("flow_matching_sumradii_centers_only", "mode", "training_and_sampling")
        else:
            _set_cfg("flow_matching_sumradii_centers_only", "mode", "retrain_and_sampling")

        console.print(f"[Pipeline] [Start training/sampling] - Iteration ({i+1}/{iterations})", style="blue")
        flow_matching_sumradii_centers_only.main(state=state)

        if not getattr(state, "samples_path", ""):
            raise RuntimeError("FM finished, but state.samples_path was not set.")
        if not getattr(state, "model_path", ""):
            raise RuntimeError("FM finished, but state.model_path was not set.")

        # Configure next push + resume path
        _set_cfg("circle_packing_SRP", "push_input", state.samples_path)
        _set_cfg("flow_matching_sumradii_centers_only", "resume_model_path", state.model_path)

        console.print(f"[Pipeline] [Start Push] - Iteration ({i+1}/{iterations})", style="blue")
        _set_cfg("circle_packing_SRP", "mode", "push_only")
        sample_generation.main(state=state)

        if not getattr(state, "pushed_samples_path", ""):
            raise RuntimeError("Push finished, but state.pushed_samples_path was not set.")

        _set_cfg("flow_matching_sumradii_centers_only", "dataset_path", state.pushed_samples_path)

    console.print("[Pipeline] Done.", style="green")
