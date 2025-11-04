# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
input_params_recovery = np.load(
    "fits/input_params_recovery_basic_lite.npy", allow_pickle=True)
input_params_recovery_em = np.load(
    "fits/input_params_recovery_em.npy", allow_pickle=True)
recovery_em = np.load("fits/recovery_em.npy", allow_pickle=True).item()
recovery_individual_mle = np.load("fits/recovery_individual_mle_basic_lite.npy",
                                  allow_pickle=True)
recovery_group_mle = np.load(
    "fits/recovery_group_mle.npy", allow_pickle=True).item()


# %%
em_recovered_params = np.stack([recovery_em['fit_participants'][i]['par_b']
                                for i in range(len(input_params_recovery_em))])

lim = [(-0.05, 1.05), (-0.05, 1.05), (-1.5, 0.05)]

for i in range(3):
    plt.figure(figsize=(4, 4))
    plt.scatter(input_params_recovery[:, i],
                em_recovered_params[:, i])
    x = np.array([float(np.ravel(a)[0])
                 for a in input_params_recovery_em[:, i]])
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

mle_recovered_params = np.stack([recovery_individual_mle[i]['par_b']
                                for i in range(len(input_params_recovery))])
n_params = 2
if n_params == 3:
    lim = [(-0.05, 1.05), (-0.05, 1.05), (-1.5, 0.05)]
elif n_params == 2:
    lim = [(-0.05, 1.05), (-1.5, 0.05)]


index = []
for i in range(len(mle_recovered_params)):
    if (np.any(mle_recovered_params[i, :] == 0) and
            np.all(input_params_recovery[i, :] != 0)):
        index.append(i)
final_result = np.delete(mle_recovered_params, index, axis=0)
final_inputs = np.delete(input_params_recovery, index, axis=0)
result_recovery_trimmed = np.delete(mle_recovered_params, index, axis=0)

for i in range(n_params):
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

for i in range(n_params):
    for j in range(i+1):
        plt.figure(figsize=(4, 4))
        plt.scatter(final_result[:, i],
                    final_result[:, j])
        plt.title(f'Param {i} vs Param {j}')
# %%
fit_params = np.load("fits/fit_params_mle_beta_10.npy", allow_pickle=True)
recovered_fits = np.load("fits/recovery_fits_mle_beta_10.npy",
                         allow_pickle=True)
lim = [(-0.05, 1.05), (-0.05, 1.05), (-5, 0.05)]
recovered_fits_params = np.stack([recovered_fits[i]['par_b']
                                  for i in range(len(fit_params))])
mask = np.where(fit_params[:, 0] != 0)
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

for i in range(3):
    for j in range(i+1):
        plt.figure(figsize=(4, 4))
        plt.scatter(recovered_fits_params[mask, i],
                    recovered_fits_params[mask, j])
        plt.title(f'Param {i} vs Param {j}')
# %%
