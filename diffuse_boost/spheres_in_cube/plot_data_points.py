import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

def load_dataset(filename):
    print(f"loading {filename}")
    T = torch.load(filename)
    print(f"Shape: {T.shape} min:{T.min()}, max:{T.max()}")

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
    return {"min_dists":np.array(min_dists), 
                   "avg_dists":avg_dists}

def plot(Arrays, labels, savepath, n_bins=100, title="Min dist", xlabel="Min dist", ylabel="Frequency", plotmode="overlay", n_xticks=15):
    max_val = -np.inf
    min_val = np.inf
    for values in Arrays:
        max_val = max(max_val, max(values))
        min_val = min(min_val, min(values))
    max_val = max_val*1.1
    min_val = min_val*0.9

    bin_edges = np.linspace(min_val, max_val, n_bins+1)
    bin_widths = np.diff(bin_edges)
    bar_x_positions = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot bars
    if plotmode == "overlay":
        max_freq = 0
        for index, values in enumerate(Arrays):
            hist_data = np.histogram(values, bins=bin_edges)[0]/len(values)
            color = mpl.colors.hsv_to_rgb((index/len(Arrays), 1, 1))
            max_freq = max(max_freq, max(hist_data))
            label = "{}[{} samples]".format(labels[index], len(values))
            plt.bar(bar_x_positions, hist_data, width=bin_widths, edgecolor='black', alpha=0.4, label=label, color=color)
        plt.ylim(0, max_freq)
    else:
        datas = []
        weights = []
        labels_samples = []
        for index, values in enumerate(Arrays):
            datas.append(values)
            w = np.empty(len(values))
            w.fill(1/len(w))
            weights.append(w)
            labels_samples.append("{}[{} samples]".format(labels[index], len(values)))
        plt.hist(datas, bin_edges, weights=weights, label=labels_samples)

    def round_step(step):
        pass
    plt.legend(loc='upper right')
    #plt.xticks(np.arange(min_val, max_val, round_step((max_val-min_val)/n_xticks)))
    plt.title(title)
    plt.xlim(min_val, max_val)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    #plt.rcParams["figure.figsize"] = (18,12)
    #plt.savefig(savepath)

def plot_files_combined(files, labels, savepath, **kwargs):
    min_distances = []
    avg_distances = []
    for file in files:
        dataset = load_dataset(file)
        metr = compute_metrics(dataset)
        min_distances.append(metr["min_dists"])
        avg_distances.append(metr["avg_dists"])
    plot(Arrays=min_distances, labels=labels, savepath=os.path.join(savepath, "mindist"), title="Normalized Min Dist Frequency",xlabel="Min Dist",**kwargs)
    plot(Arrays=avg_distances, labels=labels, savepath=os.path.join(savepath, "avgdist"), title="Normalized Average Dist Frequency",xlabel="Average Dist",**kwargs)

def plot_3d(dataset, title="plot"):
    if isinstance(dataset, str):
        dataset = torch.load(dataset)
    for data in dataset:
        ax = plt.axes(projection='3d')
        xdata, ydata, zdata = data
        ax.scatter3D(xdata, ydata, zdata, c=zdata)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        plt.show()
    


if __name__ == "__main__":
    # plot_files_combined([
    #     "output/push_simulation_PESC/2025-07-07/dataset_combined_96k.pt",
    #     "output/generated_sets/generated_20250709_102713.pt",
    #     "output/generated_sets/generated_20250706_120635.pt",
    #     "output/generated_sets/generated_20250707_111228.pt",
    #     "output/generated_sets/generated_20250708_105703.pt",
    #     "output/generated_sets/generated_20250708_160214.pt",
    #     "output/fixed_gen_sets/physics_push_2025-07-06_140100.pt"
    # ], ["Input data", "fixed penality", "After Diff(Pointnet)", "After Diff 2(Transformer)", "After Diff 3(Transformer)", "After Diff 4(Trans, penalty)", "Pointnet pushed"], n_bins=300, plotmode="overlay") 
    # plot_3d("output/generated_sets/generated_20250709_102713.pt")

    # plot_files_combined([
    #     "output/push_simulation_PESC/2025-07-07/dataset_combined_96k.pt",
    #     "output/generated_sets/generated_20250709_102713.pt",
    #     "output/generated_sets/generated_20250709_164921.pt",
    #     "output/generated_sets/generated_20250708_105703.pt",
    # ], ["Input data", "fixed penality", "Fine tune", "After Diff 2(Transformer)", "After Diff 3(Transformer)"], n_bins=300, plotmode="overlay") 
    #plot_3d("output/generated_sets/flow_gen_20250710_103909.pt")


    """data_sets = ["generated_20250710_010405.pt", "generated_20250710_011148.pt",
                 "generated_20250710_012552.pt", "generated_20250710_085625.pt",
                 "generated_20250710_094621.pt", "generated_20250710_144901.pt"]
    data_sets = ["flow_gen_20250710_123547.pt", "flow_gen_20250710_141042.pt", "flow_gen_20250710_141042.pt"]

    data_sets = [f"output/generated_sets/{name}" for name in data_sets]

    for dataname in data_sets:
        data = load_dataset(dataname)
        print(f"{dataname} min: {data.min()}, max: {data.max()}")

    plot_files_combined(data_sets, data_sets, "output")"""
    current = "output/fixed_gen_sets/physics_push_2025-07-10_160016.pt"
    #plot_3d("output/for_presentation/best_300.pt", title = "Sampled with Flow matching")
    plot_files_combined(["output/push_simulation_PESC/2025-07-07/dataset_combined_96k.pt", "output/for_presentation/best_300.pt", current], 
                        ["Training data", "Samples(Flow matching)", "Pushed Samples"], "output")


    

    """f1 = "output/push_simulation_PESC/2025-07-03/dataset_combined_sym.pt"
    d1 = load_dataset(f1)
    f1_min, f1_avg = compute_metrics(d1)

    f2 = "output/generated_sets/generated_20250704_053415.pt"
    d2 = load_dataset(f2)
    f2_min, f2_avg = compute_metrics(d2)
    print("f2_min: ", min(f2_min))

    f3 = "output/fixed_gen_sets/physics_push_2025-07-06_104040.pt"
    d3 = load_dataset(f3)
    f3_min, f3_avg = compute_metrics(d3)


    plot(Arrays=[f1_min, f2_min, f3_min], labels=("Input data", "After Diff.", "Final push"))
    #plot(Arrays=[f1_min], labels=("Input data"))"""
    