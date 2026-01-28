# %%
import numpy as np
import matplotlib.pyplot as plt
import constants
import likelihoods
import helper
import numdifftools as nd

# %% data and model fits

data_to_fit_lst = np.load('fits/data_to_fit_lst.npy', allow_pickle=True)
result_fit_mle = np.load(
    "fits/fit_individual_mle_beta_2.npy", allow_pickle=True)
result_fit_params = np.array([result_fit_mle[i]['par_b']
                              for i in range(len(result_fit_mle))])
result_diag_hess = np.array([result_fit_mle[i]['hess_diag']
                            for i in range(len(result_fit_mle))])
for i in range(3):
    plt.figure(figsize=(4, 4))
    plt.hist(result_fit_params[:, i])

# %% select

discs = np.linspace(0, 1, 50)
efficacys = np.linspace(0, 1, 50)
efforts = np.linspace(-4, 0, 50)

participant = 0
data_participant = data_to_fit_lst[participant]
fit_par = result_fit_params[participant]
print(result_diag_hess[participant])


def neg_log_lik(pars):
    nllkhd = likelihoods.likelihood_basic_model(
        pars, constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA,
        constants.REWARD_SHIRK, constants.BETA, constants.THR,
        constants.STATES_NO, [[data_participant]])
    return nllkhd

# %% inspect likelihood landscape


nllkhd = [neg_log_lik([d, fit_par[1], fit_par[2]]) for d in discs]
plt.figure()
plt.plot(discs, nllkhd)
# zoom
rnge = np.linspace(fit_par[0]-0.015, fit_par[0]+0.015, 50)
nllkhd = [neg_log_lik([d, fit_par[1], fit_par[2]])
          for d in rnge]
plt.figure()
plt.plot(rnge, nllkhd)

nllkhd = [neg_log_lik([fit_par[0], e, fit_par[2]]) for e in efficacys]
plt.figure()
plt.plot(efficacys, nllkhd)
# zoom
rnge = np.linspace(fit_par[1]-0.015, fit_par[1]+0.015, 50)
nllkhd = [neg_log_lik([fit_par[0], e, fit_par[2]])
          for e in rnge]
plt.figure()
plt.plot(rnge, nllkhd)

nllkhd = [neg_log_lik([fit_par[0], fit_par[1], e]) for e in efforts]
plt.figure()
plt.plot(efforts, nllkhd)
# zoom
rnge = np.linspace(fit_par[2]-0.015, fit_par[2]+0.015, 50)
nllkhd = [neg_log_lik([fit_par[0], fit_par[1], e])
          for e in rnge]
plt.figure()
plt.plot(rnge, nllkhd)

# %% change step size for hessians

dxs = [1e-3, 1e-4, 1e-5, 1e-6]
for dx in dxs:
    print(dx)
    hessian = helper.Hess_diag(neg_log_lik, fit_par, dx=dx)
    print(1/hessian)
    hessian_nd = nd.Hessian(neg_log_lik, step=dx)(fit_par)
    print(np.linalg.inv(hessian_nd))

# automatic step
hess = nd.Hessian(neg_log_lik)
print(np.linalg.inv(hess(fit_par)))
# hessian varies with dx and the likelihood has sharp corners
# so derivative and double derivative not valid atleast for some ppts

# %%
