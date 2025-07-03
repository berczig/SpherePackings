

import os
import torch

# Concatenate the tensors in the following three symmetrized datasets:
# output/push_simulation_PESC/2025-06-26/dataset_sym.pt, output/push_simulation_PESC/2025-07-02/dataset_sym.pt and 
# output/push_simulation_PESC/2025-07-03/dataset_top_sym.pt

sym_data_1 = torch.load("output/push_simulation_PESC/2025-06-26/dataset_sym.pt")
sym_data_2 = torch.load("output/push_simulation_PESC/2025-07-02/dataset_sym.pt")
sym_data_3 = torch.load("output/push_simulation_PESC/2025-07-03/dataset_sym.pt")

# Concatenate along the first dimension (batch dimension)
combined_sym_data = torch.cat((sym_data_1, sym_data_2, sym_data_3), dim=0)  
# Save the combined symmetrized dataset
combined_sym_fn = "output/push_simulation_PESC/2025-07-03/dataset_combined_sym.pt"
torch.save(combined_sym_data, combined_sym_fn)
print(f"Saved combined symmetrized dataset with {combined_sym_data.shape[0]} samples to {combined_sym_fn}") 

