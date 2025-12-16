import os
from rich.console import Console

import diffuse_boost
from diffuse_boost import cfg

"""
Heilbronn pipeline: (TrainSetGen or Push) -> (Train+Sample) -> Push -> (Re-train+Sample) -> Push -> ...

Config sections used:

[heilbronn_SRP]
mode = training_set_gen | final_push
output_dir
final_push_output
final_push_input

[heilbronn_flow]
mode = training_and_sampling | retrain_and_sampling | sampling_only
dataset_path
resume_model_path

[heilbronn_square_pipeline]
iterations
start_at_step = start | push | train_and_sampling
"""

# --------------------------
# cfg helpers
# --------------------------
def _ensure_section(section: str):
    if not cfg.has_section(section):
        cfg.add_section(section)

def _set_cfg(section: str, key: str, value):
    _ensure_section(section)
    cfg.set(section, key, str(value))

def _get_cfg(section: str, key: str, fallback=None):
    try:
        if isinstance(fallback, bool):
            return cfg.getboolean(section, key, fallback=fallback)
        if isinstance(fallback, int):
            return cfg.getint(section, key, fallback=fallback)
        if isinstance(fallback, float):
            return cfg.getfloat(section, key, fallback=fallback)
        if fallback is None:
            return cfg.get(section, key)
        return cfg.get(section, key, fallback=fallback)
    except Exception:
        return fallback

# --------------------------
# pipeline state
# --------------------------
class PipelineState:
    model_path: str = ""
    samples_path: str = ""
    pushed_samples_path: str = ""

    def set_model_path(self, model_path: str): self.model_path = model_path
    def set_samples_path(self, samples_path: str): self.samples_path = samples_path
    def set_pushed_samples_path(self, pushed_samples_path: str): self.pushed_samples_path = pushed_samples_path

# --------------------------
# entry
# --------------------------
if __name__ == "__main__":
    # Absolute imports (your requested paths)
    from diffuse_boost.heilbronn_square import sample_generation as data_generation
    from diffuse_boost.heilbronn_square import flow_matching_heilbronn
    from diffuse_boost.heilbronn_square import plot_data_heilbronn  # optional

    console = Console()
    state = PipelineState()

    iterations = _get_cfg("heilbronn_square_pipeline", "iterations", 1)
    start_at_step = _get_cfg("heilbronn_square_pipeline", "start_at_step", "push").strip().lower()
    console.print(f"[Heilbronn Pipeline] Start step: {start_at_step}", style="blue")

    # ----------------------------------------
    # Step 0: either generate training set or push from a given input
    # ----------------------------------------
    if start_at_step == "start":
        console.print("[Heilbronn Pipeline] [Start] training_set_gen", style="blue")
        _set_cfg("heilbronn_SRP", "mode", "training_set_gen")
        data_generation.main(state=state)  # wrapper needed in sample_generation.py (see below)
        _set_cfg("heilbronn_flow", "dataset_path", state.pushed_samples_path)

    elif start_at_step == "push":
        console.print("[Heilbronn Pipeline] [Start] final_push", style="blue")
        _set_cfg("heilbronn_SRP", "mode", "final_push")
        data_generation.main(state=state)
        _set_cfg("heilbronn_flow", "dataset_path", state.pushed_samples_path)

    elif start_at_step == "train_and_sampling":
        console.print("[Heilbronn Pipeline] [Start] train_and_sampling (expects heilbronn_flow.dataset_path already set)", style="blue")
    else:
        raise ValueError("start_at_step must be one of: start | push | train_and_sampling")

    # ----------------------------------------
    # Main loop
    # ----------------------------------------
    for i in range(iterations):
        console.print(f"[Heilbronn Pipeline] Iteration ({i+1}/{iterations})", style="blue")

        # Train + sample
        if i == 0 and start_at_step in {"start", "push", "train_and_sampling"}:
            # First iteration after dataset creation/push trains a fresh model.
            # (Resume only makes sense once we have a model from a previous iteration.)
            _set_cfg("heilbronn_flow", "mode", "training_and_sampling")
            _set_cfg("heilbronn_flow", "resume_model_path", "")
        else:
            _set_cfg("heilbronn_flow", "mode", "retrain_and_sampling")

        console.print(f"[Heilbronn Pipeline] [Train+Sample] Iteration ({i+1}/{iterations})", style="blue")
        flow_matching_heilbronn.main(state=state)  # wrapper needed in flow_matching_heilbronn.py

        # Wire outputs -> push
        _set_cfg("heilbronn_SRP", "final_push_input", state.samples_path)
        _set_cfg("heilbronn_flow", "resume_model_path", state.model_path)

        # Push samples
        console.print(f"[Heilbronn Pipeline] [Push] Iteration ({i+1}/{iterations})", style="blue")
        _set_cfg("heilbronn_SRP", "mode", "final_push")
        data_generation.main(state=state)

        # New pushed dataset becomes the next training dataset
        _set_cfg("heilbronn_flow", "dataset_path", state.pushed_samples_path)

        # Optional plotting step if you want it here
        # console.print(f"[Heilbronn Pipeline] [Plot] Iteration ({i+1}/{iterations})", style="blue")
        # plot_data_heilbronn.main()
