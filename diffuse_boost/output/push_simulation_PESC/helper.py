import os
import numpy as np
import torch
from diffuse_boost.spheres_in_cube.plot_data_points import compute_metrics
from diffuse_boost.spheres_in_cube.data_generation_PESC_symmetries import apply_symmetries_to_data
from diffuse_boost import cfg, reffolder

# Concatenate the tensors in the following three symmetrized datasets:
# output/push_simulation_PESC/2025-06-26/dataset_sym.pt, output/push_simulation_PESC/2025-07-02/dataset_sym.pt and 
# output/push_simulation_PESC/2025-07-03/dataset_top_sym.pt

#sym_data_1 = torch.load("output/push_simulation_PESC/2025-06-26/dataset_sym.pt")
#sym_data_2 = torch.load("output/push_simulation_PESC/2025-07-02/dataset_sym.pt")
#sym_data_3 = torch.load("output/push_simulation_PESC/2025-07-03/dataset_sym.pt")

def concatenate(files, output_filename):
    datasets = [torch.load(file) for file in files]
    dataset_combined = torch.cat(datasets, dim=0)
    torch.save(dataset_combined, output_filename)
    print(f"Saved combined dataset with {dataset_combined.shape[0]} samples to {output_filename}") 

def pick_best(file, output_filename, top, percentage=True):
    # top € [0,1]
    dataset = torch.load(file)
    n = dataset.shape[0]
    m = top
    if percentage:
        m = int(n*top)
    metrics = compute_metrics(dataset)
    metrics_sorted = sorted([(metrics["min_dists"][index], index) for index in range(n)], key=lambda x: -x[0])
    indices = torch.tensor([pair[1] for pair in metrics_sorted[:m]])

    output_dataset = torch.index_select(dataset, 0, indices)
    torch.save(output_dataset, output_filename)
    print(f"Picked the best {m} from {n} samples. Saved as {output_filename}")

def random_dataset(batches, d, n_spheres, filename):
    data = torch.zeros((batches, d, n_spheres))
    radius = cfg.getfloat("sample_generation_PESC", "best_known_radius")
    data.uniform_(radius, 1-radius)
    print(f"saved {batches} random samples to {filename}")
    torch.save(data, filename)

def symmetrize(filename, bounding_box_width, output_filename):
    dataset = torch.load(filename)
    print(f"input dataset: {dataset.shape}")
    sym_data = torch.from_numpy(apply_symmetries_to_data(dataset.numpy(), bounding_box_width))
    print(f"saved symmetrized data {sym_data.shape}")
    torch.save(sym_data, output_filename)

if __name__ == "__main__":
    mode = "symmetrize"
    if mode == "pick":
        #pick_best("output/push_simulation_PESC/2025-07-07/dataset_combined_96k.pt", "output/generated_sets/96k_best.pt", 0.2)
        pick_best("diffuse_boost/output/push_simulation_PP+PBTS/combined/dataset_20000.pt", "diffuse_boost/output/push_simulation_PP+PBTS/combined/dataset_20000_best_5000.pt", 5000, False)
    elif mode == "combine":
        concatenate(files = [
            "diffuse_boost/output/push_simulation_PP+PBTS/2025-08-13_09-15-22/dataset.pt",
            "diffuse_boost/output/push_simulation_PP+PBTS/2025-08-13_12-51-28/dataset.pt"],
            output_filename="diffuse_boost/output/push_simulation_PP+PBTS/combined/dataset_20000.pt")
    elif mode == "generate":
        random_dataset(10000, 3, 89, "diffuse_boost/output/generated_sets/random_dataset.pt")
    elif mode == "symmetrize":
        symmetrize(os.path.join(reffolder, "diffuse_boost/output/push_simulation_PP+PBTS/combined/dataset_5000_from_20000.pt"), 
                   8.78968670811599928, 
                   os.path.join(reffolder, "diffuse_boost/output/push_simulation_PP+PBTS/combined/dataset_5000_from_20000_sym.pt"))
