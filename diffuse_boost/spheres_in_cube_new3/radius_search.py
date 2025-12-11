"""
Outer driver for searching over sphere radii.

For each candidate radius r, this script:
1) Updates the config for sample generation and flow matching so both use that r
   (and best_known_diameter = 2r).
2) Runs the standard pipeline once.
3) Reads the final pushed metrics CSV and computes
       f(r) = 0.5 * max(post_push_min)
   which estimates the effective radius supported by the best configuration.

The helper can be used programmatically via run_radius_search or as a CLI tool.
"""

import argparse
import csv
import os
from typing import List, Tuple

from diffuse_boost.spheres_in_cube_new3.data_generation import _get_cfg, _set_cfg
from diffuse_boost.spheres_in_cube_new3.pipeline import PipelineState, main as pipeline_main


_MINSEP_COLUMNS = ("minsep", "post_push_min", "post_push_minsep", "post_min")


def _format_run_tag(r: float) -> str:
    return f"r_{r:.4f}"


def _set_radius_in_cfg(r: float) -> None:
    """Update sphere radius and best_known_diameter in all relevant config sections."""
    radius_str = f"{r:.12g}"
    diameter_str = f"{2 * r:.12g}"
    for section in ("sample_generation_PP+PBTS", "flow_matching"):
        _set_cfg(section, "sphere_radius", radius_str)
        _set_cfg(section, "best_known_diameter", diameter_str)


def _get_base_output_dirs() -> dict[tuple[str, str], str]:
    """Capture base output directories so we can append tags without accumulating them."""
    return {
        ("sample_generation_PP+PBTS", "final_push_output"): _get_cfg(
            "sample_generation_PP+PBTS", "final_push_output", ""
        ),
        ("flow_matching", "save_model_dir"): _get_cfg("flow_matching", "save_model_dir", ""),
        ("flow_matching", "save_generated_dir"): _get_cfg(
            "flow_matching", "save_generated_dir", ""
        ),
    }


def _tag_output_dirs(base_dirs: dict[tuple[str, str], str], tag: str) -> None:
    """Append a radius-specific subdirectory to key output folders."""
    for (section, key), base in base_dirs.items():
        if not base:
            continue
        tagged = os.path.join(base, tag)
        _set_cfg(section, key, tagged)


def _read_minsep_values(metrics_path: str) -> List[float]:
    """
    Extract min-separation values from a metrics CSV.
    Falls back across a few expected column names.
    """
    values: List[float] = []
    with open(metrics_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return values
        for row in reader:
            for col in _MINSEP_COLUMNS:
                if col in row and row[col] not in ("", None):
                    try:
                        values.append(float(row[col]))
                    except ValueError:
                        pass
                    break
    return values


def run_radius_search(r_values: List[float], tag_outputs: bool = True) -> List[Tuple[float, float]]:
    """
    Run the full pipeline for each candidate radius and evaluate the achieved effective radius.

    Args:
        r_values: Candidate sphere radii to evaluate.
        tag_outputs: If True, append an r-specific subdirectory to key output folders to
            avoid overwriting artifacts from different radii.

    Returns:
        List of (r, f_r) pairs, where f_r = 0.5 * max(min separation) from the final pushed set.
    """
    if not r_values:
        raise ValueError("r_values must contain at least one radius.")

    base_dirs = _get_base_output_dirs()
    results: List[Tuple[float, float]] = []

    for r in r_values:
        run_tag = _format_run_tag(r)
        print(f"\n[radius_search] === Evaluating r = {r:.6f} ===")

        _set_radius_in_cfg(r)
        if tag_outputs:
            _tag_output_dirs(base_dirs, run_tag)

        state = PipelineState()
        pipeline_main(state=state)

        metrics_path = getattr(state, "metrics_path", "")
        if not metrics_path or not os.path.exists(metrics_path):
            print(
                f"[radius_search] Warning: metrics file not found for r={r:.6f}. "
                f"Expected at '{metrics_path}'. Skipping this radius."
            )
            continue

        minsep_values = _read_minsep_values(metrics_path)
        if not minsep_values:
            print(
                f"[radius_search] Warning: no min-separation values found in {metrics_path}; "
                f"skipping r={r:.6f}."
            )
            continue

        best_minsep = max(minsep_values)
        f_r = 0.5 * best_minsep
        results.append((r, f_r))
        print(
            f"[radius_search] r={r:.6f}: best minsep={best_minsep:.6f} -> "
            f"r_eff={f_r:.6f} (metrics: {metrics_path})"
        )

    results_sorted = sorted(results, key=lambda t: t[0])
    if results_sorted:
        print("\n r        r_eff(best)")
        for r_val, f_val in results_sorted:
            print(f"{r_val:.6f}  {f_val:.6f}")
    else:
        print("\n[radius_search] No successful runs to report.")

    return results_sorted


def _parse_r_values(args: argparse.Namespace) -> List[float]:
    if args.r_values:
        return args.r_values
    if args.r_min is not None and args.r_max is not None:
        num = args.num or 1
        if num <= 0:
            raise ValueError("--num must be positive.")
        if num == 1:
            return [args.r_min]
        step = (args.r_max - args.r_min) / (num - 1)
        return [args.r_min + i * step for i in range(num)]
    raise ValueError("Provide either --r-values or --r-min/--r-max/--num.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outer search over sphere_radius using the pipeline.")
    parser.add_argument(
        "--r-values",
        type=float,
        nargs="+",
        help="Explicit list of radii to evaluate (e.g., --r-values 0.25 0.26 0.27).",
    )
    parser.add_argument("--r-min", type=float, help="Minimum radius for a linear range.")
    parser.add_argument("--r-max", type=float, help="Maximum radius for a linear range.")
    parser.add_argument("--num", type=int, default=0, help="Number of radii in the range (defaults to 1).")
    parser.add_argument(
        "--no-output-tag",
        action="store_true",
        help="Disable tagging output directories by radius (may overwrite previous runs).",
    )

    cli_args = parser.parse_args()
    radii = _parse_r_values(cli_args)
    run_radius_search(radii, tag_outputs=not cli_args.no_output_tag)
