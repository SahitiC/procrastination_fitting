# %% imports
import numpy as np
import matplotlib.pyplot as plt
import likelihoods
import gen_data
import constants
import matplotlib as mpl
import empirical_bayes
from tqdm import tqdm
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 2

# %% load data
input_params_recovery_em = np.load(
    "fits/input_params_recovery_em.npy", allow_pickle=True)
input_data_recovery_em = np.load(
    "fits/input_data_recovery_em.npy", allow_pickle=True)
recovery_em = np.load("fits/recovery_em.npy", allow_pickle=True).item()
recovery_params = np.stack([recovery_em['fit_participants'][i]['par_b']
                            for i in range(len(input_params_recovery_em))])
recovery_mle = np.load("fits/recovery_individual_mle.npy", allow_pickle=True)
recovery_params_mle = np.stack([recovery_mle[i]['par_b']
                                for i in range(len(input_params_recovery_em))])
recovery_group_mle = np.load(
    "fits/recovery_group_mle.npy", allow_pickle=True).item()

# %% is likelihood flat
np.random.seed(0)

idx = 8

# plot trajectories
# true params
plt.plot(input_data_recovery_em[idx, :][0], linewidth=2, color='black')
discount_factor_in, efficacy_in, effort_work_in = input_params_recovery_em[idx, :]
data = gen_data.gen_data_basic(
    constants.STATES, constants.ACTIONS,  constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA, discount_factor_in, efficacy_in,
    effort_work_in, 10, constants.THR, constants.STATES_NO)
for i in range(10):
    plt.plot(data[i], linestyle='dashed', color='gray')

# recovered params
plt.figure()
plt.plot(input_data_recovery_em[idx, :][0], linewidth=2, color='black')
discount_factor_rec, efficacy_rec, effort_work_rec = recovery_params[idx, :]
data = gen_data.gen_data_basic(
    constants.STATES, constants.ACTIONS,  constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA, discount_factor_rec, efficacy_rec,
    effort_work_rec, 10, constants.THR, constants.STATES_NO)
for i in range(10):
    plt.plot(data[i], linestyle='dashed', color='gray')

# plot likelihood varying discount factor
discount_factors = np.linspace(0, 1, 50)
traj = input_data_recovery_em[idx, :][0]
nllkhd_df = [likelihoods.likelihood_basic_model(
    [df, efficacy_rec, effort_work_rec], constants.STATES,
    constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA, constants.THR,
    constants.STATES_NO, traj) for df in discount_factors]
plt.figure()
plt.plot(discount_factors, nllkhd_df)
plt.xlabel('discount factor')

discount_factors = np.linspace(0, 1, 50)
traj = input_data_recovery_em[idx, :][0]
nllkhd_df = [likelihoods.likelihood_basic_model(
    [df, efficacy_in, effort_work_in], constants.STATES,
    constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA, constants.THR,
    constants.STATES_NO, traj) for df in discount_factors]
plt.figure()
plt.plot(discount_factors, nllkhd_df)
plt.xlabel('discount factor')

efficacy = np.linspace(0, 1, 50)
traj = input_data_recovery_em[idx, :][0]
nllkhd_eff = [likelihoods.likelihood_basic_model(
    [discount_factor_in, eff, effort_work_in], constants.STATES,
    constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA, constants.THR,
    constants.STATES_NO, traj) for eff in efficacy]
plt.figure()
plt.plot(efficacy, nllkhd_eff)
plt.xlabel('efficacy')

efforts = np.linspace(-4, 0, 50)
traj = input_data_recovery_em[idx, :][0]
nllkhd_efft = [likelihoods.likelihood_basic_model(
    [discount_factor_in, efficacy_in, efft], constants.STATES,
    constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA, constants.THR,
    constants.STATES_NO, traj) for efft in efforts]
plt.figure()
plt.plot(efforts, nllkhd_efft)
plt.xlabel('effort work')

print(f'true params {input_params_recovery_em[idx, :]}')
print(f'recovered params em {recovery_params[idx, :]}')
print(f'recovered params mle {recovery_params_mle[idx, :]}')

# %% BICs

# iBIC EM
pop_means = recovery_em['pop_means']
pop_vars = recovery_em['pop_vars']
samples = 100
param_ranges = empirical_bayes.get_param_ranges('basic')

nllkhd = 0
nllkhd_is = []
for i in tqdm(range(len(input_data_recovery_em))):

    params = np.random.multivariate_normal(pop_means, np.diag(pop_vars),
                                           samples)
    nllkhd_i = 0
    for s in range(samples):

        params_b = empirical_bayes.trans_to_bounded(params[s, :], param_ranges)

        nl = likelihoods.likelihood_basic_model(
            params_b, constants.STATES,
            constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, constants.THR,
            constants.STATES_NO, input_data_recovery_em[i, :][0])
        if not np.isinf(nl):
            nllkhd_i += nl
    print(nllkhd_i)
    nllkhd_is.append(nllkhd_i)
    nllkhd += (nllkhd_i / samples)
    print(nllkhd)

n_pars = 3  # 3 parameters for basic model
data_samples = len(input_data_recovery_em)
iBIC_em = 2*nllkhd + (2 * n_pars * np.log(data_samples*constants.HORIZON))

nllkhd_mle = sum([recovery_mle[i]['neg_log_lik']
                  for i in range(data_samples)])
BIC_mle = 2*nllkhd_mle + \
    (data_samples * n_pars * np.log(constants.HORIZON))

# %%
