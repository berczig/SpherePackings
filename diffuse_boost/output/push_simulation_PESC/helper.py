import os
import numpy as np
import torch
from diffuse_boost.spheres_in_cube.plot_data_points import compute_metrics

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

def pick_best(file, output_filename, top):
    # top € [0,1]
    dataset = torch.load(file)
    n = dataset.shape[0]
    m = int(n*top)
    metrics = compute_metrics(dataset)
    metrics_sorted = sorted([(metrics["min_dists"][index], index) for index in range(n)], key=lambda x: -x[0])
    indices = torch.tensor([pair[1] for pair in metrics_sorted[:m]])

    output_dataset = torch.index_select(dataset, 0, indices)
    torch.save(output_dataset, output_filename)
    print(f"Picked the best {m} from {n} samples. Saved as {output_filename}")

if __name__ == "__main__":
    mode = "pick"
    if mode == "pick":
        #pick_best("output/push_simulation_PESC/2025-07-07/dataset_combined_96k.pt", "output/generated_sets/96k_best.pt", 0.2)
        pick_best("output/generated_sets/generated_20250708_143518.pt", "output/generated_sets/transformer_300_best.pt", 0.03)
    else:
        concatenate(files = [
            "output/push_simulation_PESC/2025-07-03/dataset_combined_sym.pt",
            "output/push_simulation_PESC/2025-07-04/dataset_sym.pt"],
            output_filename="output/push_simulation_PESC/2025-07-07/dataset_combined_96k.pt")
