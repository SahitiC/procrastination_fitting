# %% imports
import numpy as np
import matplotlib.pyplot as plt
import likelihoods
import gen_data
import constants

# %% load data
input_params_recovery_em = np.load(
    "fits/input_params_recovery_em.npy", allow_pickle=True)
input_data_recovery_em = np.load(
    "fits/input_data_recovery_em.npy", allow_pickle=True)
recovery_em = np.load("fits/recovery_em.npy", allow_pickle=True).item()
recovery_params = np.stack([recovery_em['fit_participants'][i]['par_b']
                            for i in range(len(input_params_recovery_em))])

# %% is likelihood flat
np.random.seed(0)

idx = 19

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

print(f'true params {input_params_recovery_em[idx, :]}')
print(f'recovered params {recovery_params[idx, :]}')

# %%
