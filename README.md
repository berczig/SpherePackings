# DiffuseBoost: Learning Dense Sphere Packings

## Introduction

**DiffuseBoost** is a machine learning pipeline designed to find dense sphere packings in Euclidean space $\mathbb{R}^d$ and in the unit cube.

The approach aims to build a diffusion-based version of the **PatternBoost** method introduced by Charton–Ellenberg–Wagner–Williamson: [arXiv:2411.00566](https://arxiv.org/abs/2411.00566).

## Problem A: Dense Packings in $\mathbb{R}^d$

This classical problem in geometry seeks the highest possible asymptotic density of non-overlapping equal spheres in $\mathbb{R}^d$. Best known packings are currently available only in dimensions 1, 2, 3, 8, and 24.

Machine learning approaches are challenged by the need to approximate asymptotic density using large domains containing a huge number of spheres.

## Problem B: Dense Packings in the Unit Cube

This variant is well-established and comes with benchmark lower bounds. The objective is to place a fixed number of non-overlapping equal spheres in the unit cube while maximizing their radius.

See `spheres_in_cube/best_known_radius_list.txt` for known best lower bounds.

## The DiffuseBoost Pipeline

Our pipeline adapts the PatternBoost approach into a diffusion-based framework:

1. **Local Search (Physics Push)**  
   Generate a large number of candidate packings with minimal overlap using a deterministic physics push algorithm. Overlapping spheres are iteratively repelled until non-overlapping. See animations in `output/push_tests`. The top 10% of packings (measured by minimum inter-sphere distance) form the training set.

2. **Global Generation (Diffusion Model)**  
   Train a diffusion or flow-based generative model on the physics-pushed training set.  
   Sample new packings starting from white noise and denoise them using the learned model.

3. **Final Refinement (Physics Push)**  
   Apply the physics push again to remove any remaining overlaps in the generated configurations.

4. **Repeat**  
   Evaluate the distribution of minimal distances in the refined samples. Use this feedback to iterate the pipeline, enlarging or improving the training set and model.

## Repository Structure

- **`spheres_in_cube/`**  
  Implements the full pipeline for cube packings:
  - Physics push
  - Data generation
  - Diffusion/flow model training and sampling
  - Final refinement
  - Plotting tools  
  All parameters are controlled via `config.cfg`.

- **`spheres_in_Rd/`**  
  Focuses on packings in $\mathbb{R}^5$ within a box of side length 10.  
  Includes partial implementations of lattice constructions (D₅, Q₅, R₅) for comparison.  
  This component is still under active development.

- **`output/`**  
  Contains output files and visualizations for cube packings, including animations from physics push and plotted results.

- **`output_Rd/`**  
  Shows output files for $\mathbb{R}^d$ packings (primarily preliminary simulations).

- **`pipeline/`**  
  **(IGNORE for now)** Codebase for unifying the end-to-end pipeline. Under construction.

## Usage

To run the pipeline for cube packings, follow these steps:

1. Generate candidate packings using the physics push method implemented in `spheres_in_cube/data_generation_PESC.py`.

2. Train and sample from a flow or diffusion model using `spheres_in_cube/Flow_Matching_PESC.py`.

3. Apply a final physics push to remove any residual overlaps using `spheres_in_cube/final_physics_push.py`.

4. Visualize the results with `spheres_in_cube/plot_data_points.py`.

## Initial Results

Initial experimental outputs and visualizations are available in:

