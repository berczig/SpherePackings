import torch

def main():
    path = "diffuse_boost/output/generated_sets/flow_gen_20250819_141233.pt"
    data = torch.load(path)

    print(f"Loaded tensor from {path}")
    print(f"Shape: {data.shape}")
    print("Samples:\n", data)

if __name__ == "__main__":
    main()
