# %%
import numpy as np
from scipy.optimize import minimize
import likelihoods
import constants
import gen_data
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# %%


def get_num_params(model_name):
    model_params = {
        'basic_lite': 2,
        'basic': 3,
        'efficacy_gap': 4,
        'convex_concave': 4,
        'immediate_basic': 4,
        'diff_discounts': 4,
        'no_commitment': 4
    }
    return model_params[model_name]


def get_param_ranges(model_name):
    param_ranges_dict = {
        'basic_lite': [(0, 1), (None, 0)],
        'basic': [(0, 1), (0, 1), (None, 0)],
        'efficacy_gap': [(0, 1), (0, 1), (0, 1), (None, 0)],
        'convex_concave': [(0, 1), (0, 1), (None, 0), (0, None)],
        'immediate_basic': [(0, 1), (0, 1), (None, 0), (0, None)],
        'diff_discounts': [(0, 1), (0, 1), (0, 1), (None, 0)],
        'no_commitment': [(0, 1), (0, 1), (None, 0), (0, None)]
    }
    return param_ranges_dict[model_name]


def compute_log_likelihood(params, data, model_name, reward_extra):
    """Compute the log likelihood for a given model and data."""

    if model_name == 'basic_lite':
        nllkhd = likelihoods.likelihood_basic_lite(
            params, constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, constants.EFFICACY,
            constants.THR, constants.STATES_NO, data)

    elif model_name == 'basic':
        nllkhd = likelihoods.likelihood_basic_model(
            params, constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, reward_extra,
            constants.REWARD_SHIRK, constants.BETA, constants.THR,
            constants.STATES_NO, data)

    return nllkhd


def sample_initial_params(model_name, num_samples=1):
    """Sample initial parameters for MAP estimation."""

    if model_name == 'basic_lite':
        discount_factor = np.random.uniform(0.0, 1)
        effort_work = -1 * np.random.exponential(0.5)
        pars = [discount_factor, effort_work]

    elif model_name == 'basic':
        discount_factor = np.random.uniform(0.0, 1)
        efficacy = np.random.uniform(0.0, 1)
        effort_work = -1 * np.random.exponential(0.5)
        pars = [discount_factor, efficacy, effort_work]

    return pars


def sample_params(model_name, num_samples=1):
    """Sample parameters to generate data."""

    if model_name == 'basic_lite':
        discount_factor = np.random.uniform(0.2, 1)
        effort_work = -1 * np.random.exponential(0.5)
        pars = [discount_factor, effort_work]

    elif model_name == 'basic':
        discount_factor = np.random.uniform(0.2, 1)
        efficacy = np.random.uniform(0.35, 1)
        effort_work = -1 * np.random.exponential(0.5)
        pars = [discount_factor, efficacy, effort_work]

    return pars


def MLE(data_participant, model_name, iters=5, initial_guess=None):
    """
    Maximim Likelihood estimate (MLE) for a single participant.

    Parameters:
    - data_participant: array-like, shape (n_observations,)
        The input data for the participant.
    - model_name: str
        The name of the model to use ('gaussian', 'bernoulli', etc.).

    Returns:
    - fit_participant: dict
        A dictionary containing the estimated parameters and diagnostics.
    """

    param_ranges = get_param_ranges(model_name)

    if np.max(data_participant) <= 15:
        reward_extra = 0.0
    else:
        reward_extra = constants.REWARD_EXTRA

    # negative log posterior
    def neg_log_lik(pars):

        neg_log_lik = compute_log_likelihood(
            pars, data_participant, model_name, reward_extra)

        return neg_log_lik

    # optimization
    nllkhd = np.inf

    # with initial guess
    if initial_guess is not None:
        pars = initial_guess
        res = minimize(neg_log_lik, pars, bounds=param_ranges)
        if res.fun < nllkhd:
            nllkhd = res.fun
            final_res = res
        else:
            print(res.fun)

    # iterate with random initialisations
    for iter in range(iters):
        pars = sample_initial_params(model_name)
        res = minimize(neg_log_lik, pars, bounds=param_ranges)
        if res.fun < nllkhd:
            nllkhd = res.fun
            final_res = res

    fit_participant = {'par_b': final_res.x,
                       'neg_log_lik': final_res.fun,
                       'success': final_res.success}

    return fit_participant


# %%
if __name__ == "__main__":
    # %% fit models using MLE
    np.random.seed(0)

    n_participants = 150
    n_trials = 1
    paralellise = True
    data = []
    input_params = []
    parallelise = False

    # data = np.load('input_data_recovery_em.npy', allow_pickle=True)

    for i in range(n_participants):
        [discount_factor, efficacy, effort_work] = sample_params(
            'basic')
        datum = gen_data.gen_data_basic(
            constants.STATES, constants.ACTIONS,  constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, discount_factor, efficacy,
            effort_work, n_trials, constants.THR, constants.STATES_NO)
        data.append(datum)
        input_params.append([discount_factor, efficacy, effort_work])

    def fit_single_mle(datum):
        return MLE(datum, model_name='basic', iters=30)

    if parallelise:
        with ProcessPoolExecutor() as executor:
            fit_participants = list(tqdm(
                executor.map(fit_single_mle, data)))
    else:
        fit_participants = []
        for i in tqdm(range(n_participants)):
            fit_participant = MLE(data[i], model_name='basic', iters=5)
            fit_participants.append(fit_participant)

    data = np.array(data, dtype=object)
    np.save('input_data_recovery.npy', data)
    input_params = np.array(input_params, dtype=object)
    np.save('input_params_recovery.npy', input_params)
    np.save("recovery_individual_mle.npy", fit_participants, allow_pickle=True)

    # # %% fit models using MLE
    # np.random.seed(0)

    # n_participants = 150
    # n_trials = 1
    # paralellise = True
    # data = []
    # input_params = []
    # parallelise = False

    # for i in range(n_participants):
    #     [discount_factor, effort_work] = sample_params(
    #         'basic_lite')
    #     datum = gen_data.gen_data_basic(
    #         constants.STATES, constants.ACTIONS,  constants.HORIZON,
    #         constants.REWARD_THR, constants.REWARD_EXTRA,
    #         constants.REWARD_SHIRK, constants.BETA, discount_factor,
    #         constants.EFFICACY, effort_work, n_trials, constants.THR,
    #         constants.STATES_NO)
    #     data.append(datum)
    #     input_params.append([discount_factor, effort_work])

    # def fit_single_mle(datum):
    #     return MLE(datum, model_name='basic_lite', iters=30)

    # if parallelise:
    #     with ProcessPoolExecutor() as executor:
    #         fit_participants = list(tqdm(
    #             executor.map(fit_single_mle, data)))
    # else:
    #     fit_participants = []
    #     for i in tqdm(range(n_participants)):
    #         fit_participant = MLE(data[i], model_name='basic_lite', iters=5)
    #         fit_participants.append(fit_participant)

    # data = np.array(data, dtype=object)
    # np.save('input_data_recovery_basic_lite.npy', data)
    # input_params = np.array(input_params, dtype=object)
    # np.save('input_params_recovery_basic_lite.npy', input_params)
    # np.save("recovery_individual_mle_basic_lite.npy",
    #         fit_participants, allow_pickle=True)
