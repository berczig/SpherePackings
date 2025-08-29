import torch

def main():
    path = "diffuse_boost/output/generated_sets/flow_gen_20250822_110043.pt"
    data = torch.load(path)

    print(f"Loaded tensor from {path}")
    print(f"Shape: {data.shape}")
    print("Samples:\n", data[:5])  # Display first 5 samples
    # Print min and max values of the tensor
    print(f"Min value: {data.min()}, Max value: {data.max()}")

if __name__ == "__main__":
    main()
