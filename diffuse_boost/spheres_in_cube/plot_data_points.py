import os, platform
# If you need the quick unblock, leave this enabled on macOS.
# Disable by exporting SPHEREPACK_DISABLE_KMP_HACK=1 in your shell.
if platform.system() == "Darwin" and os.environ.get("SPHEREPACK_DISABLE_KMP_HACK") != "1":
    # Intel notes this is an unsafe workaround; prefer the environment fix below.
    # It must be set BEFORE any library initializes OpenMP.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def load_dataset(file):
    if isinstance(file, torch.Tensor):
        return file
    return torch.load(file)


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
    training_data = "diffuse_boost/output/heilbronn_square/training_sets/heilbronn_srp_generated_2000x10_2025-10-22_120000.pt"
    gen_samples = "diffuse_boost/output/heilbronn_square/generated_sets/heilbronn_gen_500x10_20251102_112819.pt"
    pushed_samples = "diffuse_boost/output/heilbronn_square/fixed_gen_sets/heilbronn_srp_pushed_2025-11-02_113304.pt"
    if isinstance(torch.load(training_data), dict):
        training_data = torch.load(training_data)["pushed"]
    #pushed_samplesb = "diffuse_boost/output/fixed_gen_sets/srp_pushed_2025-08-29_183046.pt"
    # merge the two pushed samples
    #pushed_samples = torch.cat((torch.load(pushed_samplesa), torch.load(pushed_samplesb)), dim=0)
    plot_files_combined([training_data, gen_samples, pushed_samples], 
                        ["Test data", "Samples(Flow matching)", "SRP Pushed Samples"], "output")
    #print the maximum values of the pushed_samples plot
    pushed_samples_data = load_dataset(pushed_samples)
    pushed_samples_metrics = compute_metrics(pushed_samples_data)
    print("Max Avg Dist (SRP Pushed Samples):", np.max(pushed_samples_metrics["avg_dists"]))
    print("Max Min Dist (SRP Pushed Samples):", np.max(pushed_samples_metrics["min_dists"]))
    # print same for training_data
    training_data = load_dataset(training_data)
    training_metrics = compute_metrics(training_data)
    print("Max Avg Dist (Training Data):", np.max(training_metrics["avg_dists"]))
    print("Max Min Dist (Training Data):", np.max(training_metrics["min_dists"]))