# %% imports

import numpy as np
import pandas as pd
import ast
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.linewidth'] = 2

# %% functions


def plot_clustered_data(data, labels):
    """
    plot trajectories in each cluster given data and labels
    """

    for label in set(labels):
        plt.figure(figsize=(4, 4), dpi=300)

        for i, d in enumerate(data):

            if labels[i] == label:

                plt.plot(np.array(ast.literal_eval(d)) * 2, alpha=0.5)

        sns.despine()
        plt.xticks([0, 7, 15])
        plt.yticks([0, 11, 14, 22])
        plt.xlabel('time (weeks)')
        plt.ylabel('research units \n completed')
        plt.savefig(
            f'plots/vectors/cluster_{label}.svg',
            format='svg', dpi=300)
        plt.show()


def get_timeseries_to_cluster(row):
    """
    get trajectories consisting of normalised cumulative progress
    """
    return ast.literal_eval(row['cumulative progress normalised'])


# %%

if __name__ == "__main__":

    data_relevant = pd.read_csv('data/data_preprocessed.csv')

    timeseries_to_cluster = list(data_relevant.apply(
        get_timeseries_to_cluster, axis=1))

    # cluster trajectories using k means for a range of cluster numbers: 2-15
    # plot inertia vs cluster number
    inertia = []
    for cluster_size in range(1, 14):
        print(cluster_size+1)
        km = KMeans(n_clusters=cluster_size+1, n_init=10,
                    random_state=0)
        labels = km.fit_predict(timeseries_to_cluster)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(inertia)
    plt.xticks(np.arange(0, 13, 2), labels=np.arange(2, 15, 2))
    plt.xlabel('cluster number')
    plt.ylabel('total within-cluster \n sum of squares')
    plt.savefig(
        'plots/vectors/inertia.svg', format='svg', dpi=300)
    plt.show()

    # final k means clustering using the best cluster size
    # best cluster size = around the elbow in inertia plot (around 6, 7, 8)
    km = KMeans(n_clusters=8, n_init=10, random_state=0, verbose=True)
    labels = km.fit_predict(timeseries_to_cluster)
    data_relevant['labels'] = labels

    # plot clustered data
    plot_clustered_data(data_relevant['cumulative progress weeks'],
                        data_relevant['labels'])

    data_relevant.to_csv('data/data_clustered.csv', index=False)

# %% distance matrix

    # rename clusters according to order in paper
    old_cluster_nos = [0, 1, 2, 3, 4, 5, 6, 7]
    new_mapping = {0: 4,
                   1: 0,
                   2: 6,
                   3: 7,
                   4: 2,
                   5: 5,
                   6: 3,
                   7: 1}

    labels_new = np.array([new_mapping[label] for label in labels])

    # sort labels and corresponding trajectories by cluster membership
    sorted_indices = np.argsort(labels_new)
    timeseries_to_cluster = np.array(timeseries_to_cluster)
    sorted_timeseries = timeseries_to_cluster[sorted_indices]
    distance_mat = distance_matrix(sorted_timeseries, sorted_timeseries)

    mask = np.tri(distance_mat.shape[0], k=0)
    mask = np.where(mask == 1, 1, np.nan)
    distance_mat = mask * distance_mat

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(distance_mat, cmap='viridis')
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Euclidean distance', rotation=270, labelpad=30)
    plt.savefig(
        'plots/vectors/distance_matrix_clusters.png',
        format='png', dpi=300)
    plt.show()
