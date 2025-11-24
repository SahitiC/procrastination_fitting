# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import constants
import gen_data

# %%
input_params_recovery_mle = np.load(
    "fits/input_params_recovery_em.npy", allow_pickle=True)
input_params_recovery_em = np.load(
    "fits/input_params_recovery_em.npy", allow_pickle=True)
recovery_em = np.load("fits/recovery_em.npy",
                      allow_pickle=True).item()
recovery_individual_mle = np.load("fits/recovery_individual_mle.npy",
                                  allow_pickle=True)
recovery_group_mle = np.load(
    "fits/recovery_group_mle.npy", allow_pickle=True).item()


# %%
em_recovered_params = np.stack([recovery_em['fit_participants'][i]['par_b']
                                for i in range(len(input_params_recovery_em))])

lim = [(0.0, 1.05), (0.0, 1.05), (-1.5, 0.05)]

for i in range(3):
    plt.figure(figsize=(4, 4))
    plt.scatter(input_params_recovery_em[:, i],
                em_recovered_params[:, i])
    x = np.array([a
                 for a in input_params_recovery_em[:, i]])
    y = np.array([a for a in em_recovered_params[:, i]])
    corr = np.corrcoef(x, y)
    plt.title(f'corr = {np.round(corr[0, 1], 3)}')
    plt.plot(
        np.linspace(lim[i][0], lim[i][1], 10),
        np.linspace(lim[i][0], lim[i][1], 10),
        linewidth=1, color='black')  # x=y line
    plt.xlim(lim[i])
    plt.ylim(lim[i])
    plt.xlabel('Input Parameter')
    plt.ylabel('Recovered Parameter')

# %%

mle_recovered_params = np.stack([recovery_individual_mle[i]['par_b']
                                for i in range(len(input_params_recovery_mle))])
n_params = 3
if n_params == 3:
    lim = [(-0.05, 1.05), (-0.05, 1.05), (-2, 0.05)]
elif n_params == 2:
    lim = [(-0.05, 1.05), (-2, 0.05)]


index = []
for i in range(len(mle_recovered_params)):
    if (np.any(mle_recovered_params[i, :] == 0) and
            np.all(input_params_recovery_mle[i, :] != 0)):
        index.append(i)
final_result = np.delete(mle_recovered_params, index, axis=0)
final_inputs = np.delete(input_params_recovery_mle, index, axis=0)

if n_params == 3:
    mask = np.where(final_result[:, 2] > -10)
    final_inputs = final_inputs[mask]
    final_result = final_result[mask]

for i in range(n_params):
    plt.figure(figsize=(4, 4))
    plt.scatter(final_inputs[:, i],
                final_result[:, i])
    x = np.array([a for a in final_inputs[:, i]])
    y = np.array([a for a in final_result[:, i]])
    corr = np.corrcoef(x, y)
    plt.title(f'corr = {np.round(corr[0, 1], 3)}')
    plt.plot(
        np.linspace(lim[i][0], lim[i][1], 10),
        np.linspace(lim[i][0], lim[i][1], 10),
        linewidth=1, color='black')  # x=y line
    plt.xlim(lim[i])
    plt.ylim(lim[i])
    plt.xlabel('Input Parameter')
    plt.ylabel('Recovered Parameter')

for i in range(n_params):
    for j in range(i+1):
        plt.figure(figsize=(4, 4))
        plt.scatter(final_result[:, i],
                    final_result[:, j])
        plt.title(f'Param {i} vs Param {j}')
# %%
fit_params = np.load("fits/fit_params_mle_low_rextra.npy", allow_pickle=True)
recovered_fits = np.load("fits/recovery_fits_mle_low_rextra.npy",
                         allow_pickle=True)
n_params = 3
if n_params == 3:
    lim = [(-0.05, 1.05), (-0.05, 1.05), (-5, 0.05)]
elif n_params == 2:
    lim = [(-0.05, 1.05), (-5, 0.05)]

tolerance = [0.45, 0.45, 1.8]

recovered_fit_params = np.stack([recovered_fits[i]['par_b']
                                 for i in range(len(fit_params))])

mask = np.where(fit_params[:, 0] != 0)
idxs = []  # to exclude
for i in range(n_params):
    colors = []
    for j in range(len(fit_params)):
        if np.abs(fit_params[j, i]-recovered_fit_params[j, i]) < tolerance[i]:
            colors.append('tab:blue')
        else:
            colors.append('tab:red')
            if not (j in idxs):
                idxs.append(j)

    colors = np.array(colors)
    print(sum(np.where(colors == 'tab:red', 1, 0)))
    plt.figure(figsize=(4, 4))
    plt.scatter(fit_params[mask, i],
                recovered_fit_params[mask, i],
                c=colors[mask])
    x = np.array([a for a in fit_params[mask, i]])
    y = np.array([a for a in recovered_fit_params[mask, i]])
    corr = np.corrcoef(x, y)
    plt.title(f'corr = {np.round(corr[0, 1], 3)}')
    plt.plot(
        np.linspace(lim[i][0], lim[i][1], 10),
        np.linspace(lim[i][0], lim[i][1], 10),
        linewidth=1, color='black')  # x=y line
    plt.xlim(lim[i])
    plt.ylim(lim[i])

    plt.xlabel('Input Parameter')
    plt.ylabel('Recovered Parameter')

# for i in range(n_params):
#     for j in range(i+1):
#         plt.figure(figsize=(4, 4))
#         plt.scatter(recovered_fit_params[mask, i],
#                     recovered_fit_params[mask, j])
#         plt.title(f'Param {i} vs Param {j}')

# %%
data = np.load('fits/data_to_fit_lst.npy', allow_pickle=True)
# %%
idx = 5
data_gen = gen_data.gen_data_basic(
    constants.STATES, constants.ACTIONS,  constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA, fit_params[idx, 0],
    fit_params[idx, 1], fit_params[idx, 2], 5, constants.THR,
    constants.STATES_NO)
data_gen_recovered = gen_data.gen_data_basic(
    constants.STATES, constants.ACTIONS,  constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA,
    recovered_fit_params[idx, 0], recovered_fit_params[idx, 1],
    recovered_fit_params[idx, 2], 5, constants.THR, constants.STATES_NO)
plt.figure()
plt.plot(data[idx])
for i in range(5):
    plt.plot(data_gen[i], color='gray')
    plt.plot(data_gen_recovered[i], color='black', linestyle='dashed')

# %% save recoverable data, plots
fit_params_recoverable = np.delete(fit_params, idxs, axis=0)

np.save('fits/fit_params_mle_recoverable.npy',
        fit_params_recoverable, allow_pickle=True)

data_full = pd.read_csv('zhang_ma_data.csv', index_col=False)
data_relevant = pd.read_csv('data_preprocessed.csv', index_col=False)
data_full_filter = data_full[data_full['SUB_INDEX_194'].isin(
    data_relevant['SUB_INDEX_194'])].reset_index(drop=True)

data_full_recoverable = data_full_filter.drop(
    index=idxs).reset_index(drop=True)
data_full_recoverable.to_csv('data_recoverable.csv', index=False)

data_processed = pd.read_csv('data_preprocessed.csv', index_col=False)
data_processed_recoverable = data_processed.drop(
    index=idxs).reset_index(drop=True)
data_processed_recoverable.to_csv(
    'data_preprocessed_recoverable.csv', index=False)
