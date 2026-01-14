# %% imports
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast

# %% data


data_relevant = pd.read_csv('data_preprocessed.csv', index_col=False)

data_full = pd.read_csv('zhang_ma_data.csv',
                        index_col=False)

data_full_filter = data_full[data_full['SUB_INDEX_194'].isin(
    data_relevant['SUB_INDEX_194'])]
data_full_filter = data_full_filter.reset_index(drop=True)

result_fit_mle = np.load(
    "fits/fit_individual_mle_with_hess.npy", allow_pickle=True)

result_fit_params = np.array([result_fit_mle[i]['par_b']
                              for i in range(len(result_fit_mle))])

result_diag_hess = np.array([result_fit_mle[i]['hess_diag']
                            for i in range(len(result_fit_mle))])

# %% remove ppts with negative hess

valid_indices = np.where(np.all(result_diag_hess > 0, axis=1))[0]

data = data_full_filter.iloc[valid_indices].reset_index(drop=True)
data_weeks = data_relevant.iloc[valid_indices].reset_index(drop=True)
fit_params = result_fit_params[valid_indices]
diag_hess = result_diag_hess[valid_indices]

# %% cluster based on parameters

km = KMeans(n_clusters=4, n_init=10, random_state=0, verbose=True)
labels = km.fit_predict(fit_params)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(fit_params[:, 2], fit_params[:, 1],
           fit_params[:, 0], c=labels, s=60, cmap='viridis')

# %%
for i in range(4):
    plt.figure()
    mask = np.where(labels == i)[0]
    for idx in mask:
        traj = np.array(ast.literal_eval(
            data_weeks['cumulative progress weeks'].iloc[idx]))
        plt.plot(2 * traj, color='gray')
    plt.title(f'Cluster {i}')
# %%
