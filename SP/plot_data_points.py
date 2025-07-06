import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

def load_dataset(filename):
    print(f"loading {filename}")
    T = torch.load(filename)
    print(f"Shape: {T.shape}")

    return torch.load(filename)

def compute_metrics(tensor_data):
    N = tensor_data.shape[2]
    min_dists = []
    avg_dists = []
    for index in range(tensor_data.shape[0]):
        packing = tensor_data[index].T
        diffs = packing[:, None, :] - packing[None, :, :]
        dmat = np.linalg.norm(diffs, axis=-1)
        i1, j1 = np.triu_indices(N, k=1)
        pdist = dmat[i1, j1]
        mn = float(pdist.min())
        av = float(pdist.mean())
        #overlap_amt = best_known_diameter - mn
        min_dists.append(mn)
        avg_dists.append(av)
    return np.array(min_dists), avg_dists

def compute_metrics2(tensor_data):
    mindists = []
    for index in range(len(tensor_data)):
        print(f"{index+1}/{len(tensor_data)}")
        mindist = 99999
        P = tensor_data[index].T
        for i in range(len(P)):
            for j in range(i+1, len(P)):
                dist = np.linalg.norm(P[i]-P[j])
                if dist < mindist:
                    mindist = dist
        mindists.append(mindist)
    return mindists, []

def plot(Arrays, labels, n_bins=100, title="Min dist", xlabel="Min dist", ylabel="Frequency"):
    max_val = -np.inf
    min_val = np.inf
    for values in Arrays:
        max_val = max(max_val, max(values))
        min_val = min(min_val, min(values))

    bin_edges = np.linspace(min_val, max_val, n_bins+1)
    bin_widths = np.diff(bin_edges)
    bar_x_positions = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot bars
    max_freq = 0
    datas = []
    weights = []
    labels_samples = []
    for index, values in enumerate(Arrays):
        datas.append(values)
        w = np.empty(len(values))
        w.fill(1/len(w))
        weights.append(w)
        labels_samples.append("{}[{} samples]".format(labels[index], len(values)))
        #hist_data = np.histogram(values, bins=bin_edges)[0]/len(values)
        #max_freq = max(max_freq, max(hist_data))
        #color = mpl.colors.hsv_to_rgb((index/len(Arrays), 1, 1))
        #plt.hist()
        #plt.bar(bar_x_positions, hist_data, width=bin_widths, edgecolor='black', alpha=0.4, label=labels[index], 
                #color=color, histtype="step")
    plt.hist(datas, bin_edges, weights=weights, label=labels_samples)

    plt.legend(loc='upper right')
    plt.xticks(np.arange(min_val, max_val, (max_val-min_val)/(n_bins/10)))
    #plt.ylim(0, max_freq)
    #plt.xlim(min_val, max_val)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == "__main__":
    f1 = "output/push_simulation_PESC/2025-07-03/dataset_combined_sym.pt"
    d1 = load_dataset(f1)
    f1_min, f1_avg = compute_metrics(d1)

    f2 = "output/generated_sets/generated_20250704_053415.pt"
    d2 = load_dataset(f2)
    f2_min, f2_avg = compute_metrics(d2)
    print("f2_min: ", min(f2_min))

    f3 = "output/fixed_gen_sets/physics_push_2025-07-05_225745.pt"
    d3 = load_dataset(f3)
    f3_min, f3_avg = compute_metrics(d3)

    f4 = "output/fixed_gen_sets/physics_push_2025-07-04_134725.pt"
    d4 = load_dataset(f4)
    f4_min, f4_avg = compute_metrics(d4)

    diff = d3-d4
    for i,k in enumerate(diff):
        print(i, k)

    plot(Arrays=[f1_min, f2_min, f3_min], labels=("Input data", "After Diff.", "Final push"))
    #plot(Arrays=[f1_min], labels=("Input data"))
    