# %%
import numpy as np
from scipy.optimize import minimize
import likelihoods
import constants
import gen_data
import mle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# %%
if __name__ == "__main__":

    # %%

    params = np.load('fits/fit_params_mle.npy', allow_pickle=True)
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

    def fit_single_mle(datum):
        return mle.MLE(datum, model_name='basic', iters=30)

    if parallelise:
        with ProcessPoolExecutor() as executor:
            fit_participants = list(tqdm(
                executor.map(fit_single_mle, data)))
    else:
        fit_participants = []
        for i in tqdm(range(len(params))):
            fit_participant = mle.MLE(data[i], model_name='basic',
                                                  iters=30)
            fit_participants.append(fit_participant)

    np.save("recovery_fits_mle.npy", fit_participants, allow_pickle=True)
