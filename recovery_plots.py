# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
input_params_recovery = np.load(
    "fits/input_params_recovery_5traj.npy", allow_pickle=True)
recovery_em = np.load("fits/recovery_em_5traj.npy", allow_pickle=True).item()
recovery_individual_mle = np.load("fits/recovery_individual_mle_5traj.npy",
                                  allow_pickle=True)
recovery_group_mle = np.load(
    "fits/recovery_group_mle.npy", allow_pickle=True).item()

n_participants = len(input_params_recovery)

# %%
em_recovered_params = np.stack([recovery_em['fit_participants'][i]['par_b']
                                for i in range(n_participants)])

lim = [(-0.05, 1.05), (-0.05, 1.05), (-1.5, 0.05)]

for i in range(3):
    plt.figure(figsize=(4, 4))
    plt.scatter(input_params_recovery[:, i],
                em_recovered_params[:, i])
    x = np.array([float(np.ravel(a)[0]) for a in input_params_recovery[:, i]])
    y = np.array([float(np.ravel(a)[0]) for a in em_recovered_params[:, i]])
    corr = np.corrcoef(x, y)
    plt.title(f'corr = {corr[0, 1]}')
    plt.plot(
        np.linspace(lim[i][0], lim[i][1], 10),
        np.linspace(lim[i][0], lim[i][1], 10),
        linewidth=1, color='black')  # x=y line
    plt.xlim(lim[i])
    plt.ylim(lim[i])

# %%
lim = [(-0.05, 1.05), (-0.05, 1.05), (-1.5, 0.05)]
mle_recovered_params = np.stack([recovery_individual_mle[i]['par_b']
                                for i in range(n_participants)])
for i in range(3):
    plt.figure(figsize=(4, 4))
    plt.scatter(input_params_recovery[:, i],
                mle_recovered_params[:, i])
    x = np.array([float(np.ravel(a)[0]) for a in input_params_recovery[:, i]])
    y = np.array([float(np.ravel(a)[0]) for a in mle_recovered_params[:, i]])
    corr = np.corrcoef(x, y)
    plt.title(f'corr = {corr[0, 1]}')
    plt.plot(
        np.linspace(lim[i][0], lim[i][1], 10),
        np.linspace(lim[i][0], lim[i][1], 10),
        linewidth=1, color='black')  # x=y line
    plt.xlim(lim[i])
    plt.ylim(lim[i])
# %%
