# %% imports
%matplotlib widget
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import ast
import regression
from mpl_toolkits.mplot3d import axes3d

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

# %% data


data_relevant = pd.read_csv('data_preprocessed.csv', index_col=False)


result_fit_mle = np.load(
    "fits/fit_individual_mle_with_hess.npy", allow_pickle=True)

result_fit_params = np.array([result_fit_mle[i]['par_b']
                              for i in range(len(result_fit_mle))])

result_diag_hess = np.array([result_fit_mle[i]['hess_diag']
                            for i in range(len(result_fit_mle))])

# %% remove ppts with negative hess

valid_indices = np.where(np.all(result_diag_hess > 0, axis=1))[0]

data_weeks = data_relevant.iloc[valid_indices].reset_index(drop=True)
fit_params = result_fit_params[valid_indices]
diag_hess = result_diag_hess[valid_indices]

# # %% cluster based on parameters
# disc = fit_params[:, 0]  # discount
# efficacy = fit_params[:, 1]  # efficacy
# effort = fit_params[:, 2]  # effort

# a = np.where(disc == 1, 0.999, disc)
# disc_trans = 1/(1-a)

# params_trans = np.vstack((disc_trans, efficacy, effort)).T

# params_zscore = ((fit_params - np.mean(fit_params, axis=0)) /
#                  np.std(fit_params, axis=0))
# params_trans_zscore = ((params_trans - np.mean(params_trans, axis=0)) /
#                        np.std(params_trans, axis=0))

# params_to_cluster = params_trans_zscore
# params_to_plot = params_trans

# # %% kmeans clustering
# km = KMeans(n_clusters=4, n_init=10, random_state=42, verbose=True)
# labels_km = km.fit_predict(params_to_cluster)

# # %% hierarchical clustering
# hierclust = AgglomerativeClustering(metric="euclidean",
#                                     compute_full_tree=True,
#                                     linkage='complete',
#                                     compute_distances=True)
# fit = hierclust.fit(params_to_cluster)
# plot_dendrogram(fit, no_labels=True)

# hierclust.set_params(n_clusters=4)
# labels_hier = hierclust.fit_predict(params_to_cluster)

# # %%
# labels = labels_km
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# p = ax.scatter(params_to_plot[:, 2], params_to_plot[:, 1],
#                params_to_plot[:, 0], c=labels, s=60, cmap='viridis')
# ax.set_xlabel('effort')
# ax.set_ylabel('efficacy')
# ax.set_zlabel('1/(1-discount)')
# ax.set_box_aspect(None, zoom=0.85)
# # fig.colorbar(p)
# plt.show()


# # %%
# for i in range(4):
#     plt.figure(figsize=(4, 4))
#     mask = np.where(labels == i)[0]
#     trajs = []
#     for idx in mask:
#         traj = np.array(ast.literal_eval(
#             data_weeks['cumulative progress weeks'].iloc[idx]))
#         plt.plot(2 * traj, color='gray')
#         trajs.append(traj)
#     trajs = np.array(trajs)
#     mean_traj = np.mean(trajs, axis=0)
#     plt.title(f'Cluster {i}')
# %%
# statistics
mucw = np.array(data_weeks.apply(regression.get_mucw, axis=1))
trajectories = np.array(data_weeks['cumulative progress weeks'])
trajectories = np.array(
    [ast.literal_eval(data_weeks['cumulative progress weeks'][i])
     for i in range(len(data_weeks))])
trajectories = trajectories*2
units_completed = np.array([np.max(trajectories[i])
                            for i in range(len(trajectories))])
first_third = trajectories[:, 5]
last_third = trajectories[:, 15] - trajectories[:, 7]
time_for_halfpoint = [np.argmin(
    np.abs(trajectories[i]-np.max(trajectories[i])/2)) + 1
    for i in range(len(trajectories))]
# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', elev=20, azim=-45)
p = ax.scatter(fit_params[:, 2], fit_params[:, 1],
               fit_params[:, 0], c=units_completed, s=60, cmap='viridis')
ax.set_title('units completed')
ax.set_xlabel('effort')
ax.set_ylabel('efficacy')
ax.set_zlabel('discount')
ax.set_box_aspect(None, zoom=0.7)
fig.colorbar(p)
plt.show()

# %%
