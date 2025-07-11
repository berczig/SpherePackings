import torch

data = torch.load("/Users/au596283/MLProjects/SpherePacking/output/push_simulation/2025-05-27 15_38_12/dataset.pt")
if hasattr(data, 'shape'):
    print("Shape of dataset.pt:", data.shape)
elif isinstance(data, dict):
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"Shape of {k}:", v.shape)
else:
    print("Loaded object type:", type(data))

