import os

import numpy as np
from matplotlib import pyplot as plt

PLOT_DIR = "plots"


class PlotHelper:
    def __init__(self, list_of_label_to_count, gpu_per_node):
        self.list_of_label_to_count = list_of_label_to_count
        self.max_entropy = self.__calculate_max_entropy()
        self.gpu_per_node = gpu_per_node
        if not os.path.exists(PLOT_DIR):
            os.mkdir(PLOT_DIR)

    def histogram_raster(self, rows):
        fig, axs = plt.subplots(rows, self.gpu_per_node, sharex=True, sharey=True, tight_layout=True)
        for i, labels in enumerate(self.list_of_label_to_count):
            index_pair = (i // self.gpu_per_node, i % self.gpu_per_node) \
                if rows > 1 else i % self.gpu_per_node
            axs[index_pair].hist(labels)
            axs[index_pair].set_title(f"Rank {i}", fontsize=10)
            if i == 0:
                axs[index_pair].set_xlabel("Label")
                axs[index_pair].set_ylabel("Count")
        store_plot(filename="histogram_raster.png")

    def all_lines(self):
        plt.yscale('log')
        for label_to_count in self.list_of_label_to_count:
            plt.plot(list(label_to_count.keys()), list(label_to_count.values()))
        plt.title("Label distribution of all ranks")
        plt.xlabel("Label")
        plt.ylabel("Count")
        store_plot(filename="all_lines.png")

    def per_rank_with_entropy(self):
        SUBFOLDER = "per_rank"
        if not os.path.exists(f"{PLOT_DIR}/{SUBFOLDER}"):
            os.mkdir(f"{PLOT_DIR}/{SUBFOLDER}")
        for i, label_to_count in enumerate(self.list_of_label_to_count):
            plt.yscale('log')
            plt.plot(list(label_to_count.keys()), list(label_to_count.values()))
            entropy = calculate_entropy(label_to_count)
            plt.title(f"Relative entropy: {entropy:.3f} / {self.max_entropy:.3f} "
                      f"= {entropy / self.max_entropy:.6f}", fontsize=10)
            plt.xlabel("Label")
            plt.ylabel("Count")
            store_plot(filename=f"{SUBFOLDER}/{i}.png")

    def __calculate_max_entropy(self):
        classes = len(self.list_of_label_to_count[0].keys())
        for label_to_count in self.list_of_label_to_count:
            classes = max(classes, len(label_to_count.keys()))
        return np.log2(classes)


def calculate_entropy(label_to_count):
    total = sum(label_to_count.values())
    probabilities = [count / total for count in label_to_count.values()]
    return -sum([p * np.log2(p) for p in probabilities])


def get_label_to_count(all_labels):
    labels, counts = np.unique(all_labels, return_counts=True)
    label_to_count = dict(zip(labels, counts))
    return {k: label_to_count[k] for k in sorted(label_to_count)}


def store_plot(filename):
    plt.savefig(f"{PLOT_DIR}/{filename}")
    plt.close()
