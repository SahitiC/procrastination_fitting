# %%
import numpy as np
from scipy.optimize import minimize
import likelihoods
import constants
import gen_data
import mle2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# %%
if __name__ == "__main__":

    # # %%

    params = np.load('fit_params_mle_2_rextras.npy', allow_pickle=True)
    n_trials = 1
    data = []
    parallelise = True

    for i in range(len(params)):

        [discount_factor, efficacy, effort_work] = params[i, :]

        datum = gen_data.gen_data_basic(
            constants.STATES, constants.ACTIONS,  constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, discount_factor, efficacy,
            effort_work, n_trials, constants.THR, constants.STATES_NO)
        data.append(datum)

    def fit_single_mle(args):
        datum, initial_guess = args
        return mle2.MLE(datum, model_name='basic',  iters=50,
                       initial_guess=initial_guess)

    # def fit_single_mle(datum, initial_guess):
    #     return mle.MLE(datum, model_name='basic', iters=30)

    if parallelise:
        args_list = []  # store args
        for i in range(len(params)):
            initial_guess = params[i, :]
            args_list.append(
                (data[i], initial_guess))
        with ProcessPoolExecutor() as executor:
            fit_participants = list(tqdm(
                executor.map(fit_single_mle, args_list)))
    else:
        fit_participants = []
        for i in tqdm(range(len(params))):
            fit_participant = mle2.MLE(data[i], model_name='basic',
                                      iters=50, initial_guess=params[i, :])
            fit_participants.append(fit_participant)

    np.save("recovery_fits_mle_2_rextras.npy",
            fit_participants, allow_pickle=True)

    # %%

    # params = np.load('fit_params_mle_trunc_bounds.npy', allow_pickle=True)
    # n_trials = 1
    # data = []
    # parallelise = True

    # for i in range(len(params)):

    #     [discount_factor, effort_work] = params[i, :]

    #     datum = gen_data.gen_data_basic(
    #         constants.STATES, constants.ACTIONS,  constants.HORIZON,
    #         constants.REWARD_THR, constants.REWARD_EXTRA,
    #         constants.REWARD_SHIRK, constants.BETA, discount_factor,
    #         constants.EFFICACY, effort_work, n_trials, constants.THR,
    #         constants.STATES_NO)
    #     data.append(datum)

    # def fit_single_mle(args):
    #     datum, initial_guess = args
    #     return mle.MLE(datum, model_name='basic_lite', iters=30,
    #                    initial_guess=initial_guess)

    # if parallelise:
    #     args_list = []  # store args
    #     for i in range(len(params)):
    #         initial_guess = params[i, :]
    #         args_list.append(
    #             (data[i], initial_guess))
    #     with ProcessPoolExecutor() as executor:
    #         fit_participants = list(tqdm(
    #             executor.map(fit_single_mle, args_list)))
    # else:
    #     fit_participants = []
    #     for i in tqdm(range(len(params))):
    #         fit_participant = mle.MLE(data[i], model_name='basic_lite',
    #                                   iters=30, initial_guess=params[i, :])
    #         fit_participants.append(fit_participant)

    # np.save("recovery_fits_mle_trunc_bounds.npy",
    #         fit_participants, allow_pickle=True)

# %%
