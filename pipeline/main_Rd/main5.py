import os
import torch
import numpy as np
from datetime import datetime
from spheres_in_Rd.DiffusionModel import (
    train_diffusion_model,
    sample_diffusion_model,
    PointNetPlusPlus,
    SetTransformer,
)
from spheres_in_Rd.physics_push import eliminate_overlaps_batched
from spheres_in_Rd.data_generation import get_data_loader
from spheres_in_Rd import cfg

def main():
    # 1) Load config
    batch_size            = cfg.getint("diffusion", "batch_size")
    num_epochs            = cfg.getint("diffusion", "num_epochs")
    learning_rate         = cfg.getfloat("diffusion", "learning_rate")
    num_train_timesteps   = cfg.getint("diffusion", "num_train_timesteps")
    num_inference_timesteps = cfg.getint("diffusion", "num_inference_timesteps")
    clip_sample           = cfg.getboolean("diffusion", "clip_sample")
    clip_sample_range     = cfg.getfloat("diffusion", "clip_sample_range")
    model_type            = cfg.get("diffusion", "model_type")
    sphere_radius         = cfg.getfloat("diffusion", "sphere_radius")
    beta_start            = cfg.getfloat("diffusion", "beta_start")
    beta_end              = cfg.getfloat("diffusion", "beta_end")
    device                = torch.device(cfg.get("diffusion", "device"))
    num_new_from_one      = cfg.getint("diffusion", "num_new_from_one")
    num_loops             = cfg.getint("diffusion", "num_loops")
    # physics‐push params (to clean samples)
    n_points              = cfg.getint("physics_push", "n_points")
    dimension             = cfg.getint("physics_push", "dimension")
    box_size              = cfg.getfloat("physics_push", "box_size")
    max_iter              = cfg.getint("physics_push", "max_iter")
    evaluations           = cfg.getint("physics_push", "evaluations")
    dt                    = cfg.getfloat("physics_push", "dt")
    tol                   = cfg.getfloat("physics_push", "tol")
    eval_skip             = max_iter // evaluations
    batched_iterations    = eval_skip * np.ones(evaluations)

    # 2) Instantiate model
    if model_type.lower() == "pointnet":
        model = PointNetPlusPlus(d=dimension)
    else:
        model = SetTransformer(
            dim_in=dimension,
            dim_hidden=cfg.getint("diffusion", "st_dim_hidden"),
            num_heads=cfg.getint("diffusion", "st_num_heads"),
            num_inds=cfg.getint("diffusion", "st_num_inds"),
            num_isab=cfg.getint("diffusion", "st_num_isab"),
            dim_out=dimension,
        )
    model = model.to(device)

    # 3) Load or initialize dataset
    dataset_path = cfg.get("diffusion", "dataset_load_path")
    save_path    = cfg.get("diffusion", "dataset_save_path")
    if dataset_path and os.path.exists(dataset_path):
        dataset = torch.load(dataset_path)    # (M, d, N)
        print(f"Loaded {len(dataset)} initial samples from {dataset_path}")
    else:
        # start from random packings
        dataset = torch.empty((0, dimension, n_points), dtype=torch.float32)

    # 4) Iterative loop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for loop in range(num_loops):
        print(f"\n=== Iteration {loop+1}/{num_loops} ===")

        # a) Build DataLoader and train
        train_loader = get_data_loader(batch_size, dataset_path=None, tensor_dataset=dataset)
        model, _     = train_diffusion_model(
            model, train_loader,
            num_epochs, learning_rate,
            num_train_timesteps, sphere_radius,
            beta_start, beta_end,
            clip_sample, clip_sample_range,
            device
        )

        # b) Generate new samples
        gen_loader = get_data_loader(batch_size, dataset_path=None, tensor_dataset=dataset)
        _, _, samples = sample_diffusion_model(
            model, gen_loader,
            num_train_timesteps, num_inference_timesteps,
            beta_start, beta_end,
            clip_sample, clip_sample_range,
            device, num_new_from_one
        )
        # samples: (M_gen, k, d, N) → take final step
        cand = torch.tensor(samples[:, -1], dtype=torch.float32)  # (M_gen, d, N)

        # c) Clean with physics push & evaluate min‐dist
        metrics = []
        cleaned = []
        for s in cand:
            pts = s.cpu().numpy().T  # (N, d)
            final, evals = eliminate_overlaps_batched(
                pts, sphere_radius, box_size*np.ones(dimension),
                batched_iterations=batched_iterations, dt=dt, tol=tol, visualize=False
            )
            md = min(evals[max_iter]["min_distances"])
            metrics.append(md)
            cleaned.append(torch.tensor(final.T, dtype=torch.float32))

        metrics = np.array(metrics)
        # d) Select top 10% (largest min‐dist)
        k = max(1, int(0.1 * len(metrics)))
        idx = np.argsort(metrics)[::-1][:k]
        new_best = [cleaned[i] for i in idx]
        print(f" Selected {k} best of {len(cleaned)} (min‐dist ≥ {metrics[idx].min():.4f})")

        # e) Augment dataset and save
        aug = torch.stack(new_best, dim=0)  # (k, d, N)
        dataset = torch.cat([dataset, aug], dim=0)
        torch.save(dataset, save_path)
        print(f" Dataset now = {len(dataset)} samples (saved to {save_path})")

    # 5) Final model save
    final_name = f"diffusion_iter_{timestamp}.pth"
    final_path = os.path.join("output/saved_models", final_name)
    torch.save(model.state_dict(), final_path)
    print(f"\nFinished. Final model: {final_path}")

if __name__ == "__main__":
    main()