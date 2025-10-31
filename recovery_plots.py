# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
input_params_recovery = np.load(
    "fits/input_params_recovery.npy", allow_pickle=True)
input_params_recovery_em = np.load(
    "fits/input_params_recovery_em.npy", allow_pickle=True)
recovery_em = np.load("fits/recovery_em_10traj.npy", allow_pickle=True).item()
recovery_individual_mle = np.load("fits/recovery_individual_mle.npy",
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

index = []
for i in range(len(mle_recovered_params)):
    if (np.any(mle_recovered_params[i, :] == 0) and
            np.all(input_params_recovery[i, :] != 0)):
        index.append(i)
final_result = np.delete(mle_recovered_params, index, axis=0)
final_inputs = np.delete(input_params_recovery, index, axis=0)
result_recovery_trimmed = np.delete(mle_recovered_params, index, axis=0)

for i in range(3):
    plt.figure(figsize=(4, 4))
    plt.scatter(final_inputs[:, i],
                final_result[:, i])
    x = np.array([float(np.ravel(a)[0]) for a in final_inputs[:, i]])
    y = np.array([float(np.ravel(a)[0]) for a in final_result[:, i]])
    corr = np.corrcoef(x, y)
    plt.title(f'corr = {corr[0, 1]}')
    plt.plot(
        np.linspace(lim[i][0], lim[i][1], 10),
        np.linspace(lim[i][0], lim[i][1], 10),
        linewidth=1, color='black')  # x=y line
    plt.xlim(lim[i])
    plt.ylim(lim[i])
# %%
fit_params = np.load("fits/fit_params_mle.npy", allow_pickle=True)
recovered_fits = np.load("fits/recovery_fits_mle.npy", allow_pickle=True)
lim = [(-0.05, 1.05), (-0.05, 1.05), (-2, 0.05)]
recovered_fits_params = np.stack([recovered_fits[i]['par_b']
                                for i in range(len(fit_params))])
mask = np.where(fit_params[:, 0]!=0)
for i in range(3):
    plt.figure(figsize=(4, 4))
    plt.scatter(fit_params[mask, i],
                recovered_fits_params[mask, i])
    x = np.array([float(np.ravel(a)[0]) for a in fit_params[:, i]])
    y = np.array([float(np.ravel(a)[0]) for a in recovered_fits_params[:, i]])
    corr = np.corrcoef(x, y)
    plt.title(f'corr = {corr[0, 1]}')
    plt.plot(
        np.linspace(lim[i][0], lim[i][1], 10),
        np.linspace(lim[i][0], lim[i][1], 10),
        linewidth=1, color='black')  # x=y line
    plt.xlim(lim[i])
    plt.ylim(lim[i])
# %%
