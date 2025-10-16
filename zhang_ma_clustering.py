# %%
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import ast
from scipy.cluster.hierarchy import dendrogram
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3

# %%


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # from sklearn example code

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_clustered_data(data, labels, **kwargs):

    for label in set(labels):
        plt.figure(figsize=(5, 4), dpi=100)

        for i in range(len(data)):

            if labels[i] == label:

                plt.plot(ast.literal_eval(data[i]), alpha=0.5)

        sns.despine()
        plt.xticks([0, 7, 15])
        plt.yticks([0, 5, 11])


def silhoutte_plots(data, labels, n_clusters, **kwargs):
    """
    plot silhouette score for each sample (in a sorted order) given data
    and labels
    (code adapted from sklearn example: 
     https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

    """

    sample_silhouette_values = silhouette_samples(timeseries_to_cluster,
                                                  labels)
    y_lower = 10
    plt.figure()
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        plt.ylim([0, len(timeseries_to_cluster) + (n_clusters + 1) * 10])
        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # slihoutte score
    silhouette_avg = silhouette_score(timeseries_to_cluster, labels)
    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.ylabel('cluster label')
    plt.xlabel('silhouette value')


# %%

if __name__ == "__main__":

    # %%
    data_relevant = pd.read_csv('data/data_preprocessed.csv')

    timeseries_to_cluster = []
    for i in range(len(data_relevant)):
        timeseries_to_cluster.append((ast.literal_eval(
            data_relevant['cumulative progress'][i])))

    timeseries_to_cluster = np.array(timeseries_to_cluster)

    # %%
    # cluster sequences (unit completion times) using k means

    # inertia vs cluuster number
    inertia = []
    silhouette_scores = []
    for cluster_size in range(1, 15):
        print(cluster_size+1)
        km = KMeans(n_clusters=cluster_size+1, n_init=10,
                    random_state=0)
        labels = km.fit_predict(timeseries_to_cluster)
        inertia.append(km.inertia_)
        silhouette_scores.append(silhouette_score(timeseries_to_cluster,
                                                  labels))
        silhoutte_plots(timeseries_to_cluster, labels, cluster_size+1)

    plt.figure()
    plt.plot(inertia)
    plt.xticks(np.arange(14), labels=np.arange(2, 16))
    plt.xlabel('cluster number')
    plt.ylabel('k-means sum of squares')

    plt.figure()
    plt.plot(silhouette_scores)
    plt.xticks(np.arange(14), labels=np.arange(2, 16))
    plt.xlabel('cluster number')
    plt.ylabel('silhouette score')

    # final k means clustering using the best cluster no.
    km = KMeans(n_clusters=8, n_init=10, random_state=0, verbose=True)
    labels = km.fit_predict(timeseries_to_cluster)
    data_relevant['labels'] = labels

    # plot clustered data
    plot_clustered_data(data_relevant['cumulative progress weeks'],
                        data_relevant['labels'])

    # %%
    # agglomerative clustering
    model = AgglomerativeClustering(metric="euclidean",
                                    compute_full_tree=True,
                                    linkage='complete',
                                    compute_distances=True)
    fit = model.fit(timeseries_to_cluster)
    plot_dendrogram(fit, no_labels=True)

    model.set_params(n_clusters=4)
    clusters = model.fit_predict(timeseries_to_cluster)
    plot_clustered_data(data_relevant['cumulative progress weeks'], clusters)
