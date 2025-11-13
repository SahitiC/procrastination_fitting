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
        'rl-basic': 2,
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
        'rl-basic': [(0, 1), (0, None)],
        'basic': [(0, 1), (0, 1), (None, 0)],
        'efficacy_gap': [(0, 1), (0, 1), (0, 1), (None, 0)],
        'convex_concave': [(0, 1), (0, 1), (None, 0), (0, None)],
        'immediate_basic': [(0, 1), (0, 1), (None, 0), (0, None)],
        'diff_discounts': [(0, 1), (0, 1), (0, 1), (None, 0)],
        'no_commitment': [(0, 1), (0, 1), (None, 0), (0, None)]
    }
    return param_ranges_dict[model_name]


def compute_log_likelihood(params, data, model_name):
    """Compute the log likelihood for a given model and data."""

    if model_name == 'rl-basic':
        nllkhd = likelihoods.likelihood_rl_basic_model(
            params, data)

    elif model_name == 'basic':
        nllkhd = likelihoods.likelihood_basic_model(
            params, constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, constants.THR,
            constants.STATES_NO, data)

    return nllkhd


def sample_initial_params(model_name, num_samples=1):
    """Sample initial parameters for MAP estimation."""

    if model_name == 'rl-basic':
        alpha_u = np.random.logistic(0, 1)
        beta_u = np.random.logistic(0, 1)
        pars = [alpha_u, beta_u]

    elif model_name == 'basic':
        discount_factor = np.random.logistic(0, 1)
        efficacy = np.random.logistic(0, 1)
        effort_work = np.random.normal(loc=-1.2, scale=1.2)
        pars = [discount_factor, efficacy, effort_work]

    # elif model_name == 'basic':
    #     discount_factor = np.random.uniform(0, 1)
    #     efficacy = np.random.uniform(0, 1)
    #     effort_work = -1 * np.random.exponential(0.5)
    #     pars = [discount_factor, efficacy, effort_work]

    return pars


def sample_params(model_name, num_samples=1):
    """Sample parameters to generate data."""

    if model_name == 'rl-basic':
        alpha_u = np.random.uniform(0.2, 0.9)
        beta_u = np.random.uniform(0.5, 5)
        pars = [alpha_u, beta_u]

    elif model_name == 'basic':
        discount_factor = np.random.uniform(0.2, 1)
        efficacy = np.random.uniform(0.35, 1)
        effort_work = -1 * np.random.exponential(0.5)
        pars = [discount_factor, efficacy, effort_work]

    return pars


def Hess_diag(fun, x, dx=1e-4):
    """Evaluate the diagonal elements of the hessian matrix using the 3 point
    central difference formula with spacing of dx between points."""
    n = len(x)
    hessdiag = np.zeros(n)
    for i in range(n):
        dx_i = np.zeros(n)
        dx_i[i] = dx
        hessdiag[i] = (fun(x + dx_i) + fun(x - dx_i) -
                       2. * fun(x)) / (dx ** 2.)
    return hessdiag


def trans_to_bounded(pars, param_ranges):
    """Transform parameters to be within bounds using a sigmoid function."""
    bounded_pars = np.zeros_like(pars)
    for i, (low, high) in enumerate(param_ranges):
        if low is None and high is None:
            bounded_pars[i] = pars[i]
        elif low is None and high == 0:
            bounded_pars[i] = high - np.exp(pars[i])
        elif low == 0 and high is None:
            bounded_pars[i] = low + np.exp(pars[i])
        else:
            if pars[i] < -100.:
                pars[i] = -100.
            bounded_pars[i] = low + (high - low) / (1 + np.exp(-pars[i]))
    return bounded_pars


def trans_to_unbounded(pars_bounded, param_ranges):
    """Transform bounded parameters back to the unconstrained space."""
    unbounded_pars = np.zeros_like(pars_bounded)
    for i, (low, high) in enumerate(param_ranges):
        x = pars_bounded[i]
        if low is None and high is None:
            unbounded_pars[i] = x
        elif low is None and high == 0:
            unbounded_pars[i] = np.log(high - x)
        elif low == 0 and high is None:
            unbounded_pars[i] = np.log(x - low)
        else:
            ratio = (x - low) / (high - low)
            # clip to avoid log(0)
            ratio = np.clip(ratio, 1e-9, 1 - 1e-9)
            unbounded_pars[i] = np.log(ratio / (1 - ratio))
    return unbounded_pars


def MAP(data_participant, model_name, pop_means=None,
        pop_vars=None, iters=5, only_mle=False, initial_guess=None):
    """
    Maximum a posteriori (MAP) estimation for a single participant.

    Parameters:
    - data_participant: array-like, shape (n_observations,)
        The input data for the participant.
    - pop_mean: array-like, shape (n_params,)
        The population mean parameters.
    - pop_var: array-like, shape (n_params,)
        The population variance parameters.
    - model_name: str
        The name of the model to use ('gaussian', 'bernoulli', etc.).

    Returns:
    - fit_participant: dict
        A dictionary containing the estimated parameters and diagnostics.
    """

    param_ranges = get_param_ranges(model_name)

    # negative log posterior
    def neg_log_post(pars):

        pars_bounded = trans_to_bounded(pars, param_ranges)
        log_lik = compute_log_likelihood(
            pars_bounded, data_participant, model_name)

        if only_mle:
            return log_lik
        else:
            log_prior = - (len(pars) / 2.) * np.log(2 * np.pi) - np.sum(
                np.log(pop_vars)) / 2. - sum((pars - pop_means)
                                             ** 2. / (2 * pop_vars))
            return (log_lik - log_prior)

    # optimization
    post = np.inf

    # with initial guess
    if initial_guess is not None:
        pars = initial_guess
        valid_fit_found = False
        res = minimize(neg_log_post, pars)
        diag_hess = Hess_diag(neg_log_post, res.x)
        if min(diag_hess) > 0:
            valid_fit_found = True
        if not valid_fit_found:
            diag_hess[diag_hess < 0] = 1/6.25  # prior variance
        if res.fun < post:
            post = res.fun
            final_res = res
            diag_hess_final = diag_hess
        else:
            print(res.fun, pop_means, pop_vars)

    # iterate with random initialisations
    for iter in range(iters):
        pars = sample_initial_params(model_name)
        valid_fit_found = False
        res = minimize(neg_log_post, pars)
        diag_hess = Hess_diag(neg_log_post, res.x)
        if min(diag_hess) > 0:
            valid_fit_found = True
        if not valid_fit_found:
            diag_hess[diag_hess < 0] = 1/6.25  # prior variance
        if res.fun < post:
            post = res.fun
            final_res = res
            diag_hess_final = diag_hess

    par_b = trans_to_bounded(final_res.x, param_ranges)

    fit_participant = {'par_b': par_b,  # bounded params
                       'par_u': final_res.x,  # unbounded params
                       'diag_hess': diag_hess_final,
                       'neg_log_post': final_res.fun,
                       'success': final_res.success}

    return fit_participant


def fit_single(args):
    datum, model_name, pop_means, pop_vars, initial_guess = args
    return MAP(datum, model_name, pop_means, pop_vars,
               initial_guess=initial_guess)


def em(data, model_name, max_iter=20, tol=1e-3, parallelise=False):
    """
    Run the EM algorithm for a given model and data.

    Parameters:
    - data: array-like, shape (n_participants,)
        The input data for the EM algorithm.
    - model_name: str
        The name of the model to use ('basic', 'efficacy_gap', etc.)
    - max_iter: int, optional (default=20)
        The maximum number of iterations to run the EM algorithm.
    - tol: float, optional (default=1e-3)
        The tolerance for convergence.

    Returns:
    - params: dict
        The estimated parameters of the model.
    """

    n_params = get_num_params(model_name)
    param_ranges = get_param_ranges(model_name)

    # initialise prior
    pop_means = np.zeros(n_params)  # or np.random.randn(n_params)
    pop_vars = np.ones(n_params) * 100
    total_llkhd = 0

    num_participants = len(data)

    # EM algorithm

    for iteration in tqdm(range(max_iter)):

        # E-step
        fit_participants = []
        history_total_llkhd = []

        if parallelise:
            args_list = []  # store args
            for i in range(num_participants):
                # initial guess from previous iteration
                initial_guess = (old_participant_fits[i]
                                 if iteration > 0 else None)
                args_list.append(
                    (data[i], model_name, pop_means, pop_vars, initial_guess))
            with ProcessPoolExecutor() as executor:
                fit_participants = list(executor.map(fit_single, args_list))
        else:
            for i in range(num_participants):
                # initial guess from previous iteration
                initial_guess = (old_participant_fits[i]
                                 if iteration > 0 else None)
                fit_participant = MAP(data[i], model_name, pop_means,
                                      pop_vars, initial_guess=initial_guess)
                fit_participants.append(fit_participant)

        # M-step
        pars_U = np.array([fit_participant['par_u']
                           for fit_participant in fit_participants])
        diag_hess = np.array([fit_participant['diag_hess']
                              for fit_participant in fit_participants])
        new_pop_means = np.mean(pars_U, axis=0)
        new_pop_vars = np.mean(pars_U**2.+1./diag_hess,
                               axis=0)-new_pop_means**2.
        new_total_llkhd = compute_log_likelihood(
            trans_to_bounded(new_pop_means, param_ranges), data, model_name)

        print(np.abs(new_pop_means-pop_means), np.abs(new_pop_vars-pop_vars))
        print(f'diff in llkhd: {new_total_llkhd - total_llkhd}')
        history_total_llkhd.append(total_llkhd)

        # check convergence
        if np.max(np.abs(new_pop_means-pop_means)) < tol and np.max(
                np.abs(new_pop_vars-pop_vars)) < tol:

            print(f'Converged in {iteration} iterations.')
            pop_means = new_pop_means
            pop_vars = new_pop_vars
            total_llkhd = new_total_llkhd
            break

        pop_means = new_pop_means
        pop_vars = new_pop_vars
        total_llkhd = new_total_llkhd

        old_participant_fits = [fit_participants[i]['par_u']
                                for i in range(num_participants)]

    fit_pop = {'pop_means': pop_means, 'pop_vars': pop_vars,
               'fit_participants': fit_participants, 'model_name': model_name,
               'history_llkhd': history_total_llkhd}

    return fit_pop


# %%
if __name__ == "__main__":
    # %% fit models using Emirical Bayes

    np.random.seed(0)

    n_participants = 150
    n_trials = 1
    paralellise = True
    data = []
    input_params = []
    param_ranges = get_param_ranges('basic')
    means = [2.905,  1.546, -1.221]  # sample means
    vars = np.array([0.010, 0.023, 0.301])  # sample variances
    samples = np.random.multivariate_normal(
        means, np.diag(np.sqrt(vars)), n_participants)
    for i in range(n_participants):

        [discount_factor, efficacy, effort_work] = trans_to_bounded(
            samples[i, :], param_ranges)
        datum = gen_data.gen_data_basic(
            constants.STATES, constants.ACTIONS,  constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, discount_factor, efficacy,
            effort_work, n_trials, constants.THR, constants.STATES_NO)
        data.append(datum)
        input_params.append([discount_factor, efficacy, effort_work])

    fit_pop = em(data, model_name='basic', max_iter=20, tol=0.01,
                 parallelise=paralellise)
    print(fit_pop)
    np.save("recovery_em_dist.npy", fit_pop, allow_pickle=True)

    data = np.array(data, dtype=object)
    np.save('input_data_recovery_em_dist.npy', data)
    input_params = np.array(input_params, dtype=object)
    np.save('input_params_recovery_em_dist.npy', input_params)

    # %% run MLE for individuals

    # def fit_single_mle(datum):
    #     return MAP(datum, model_name='basic', iters=20, only_mle=True)
    # if paralellise:
    #     with ProcessPoolExecutor() as executor:
    #         fit_participants = list(tqdm(
    #             executor.map(fit_single_mle, data)))
    # else:
    #     fit_participants = []
    #     for i in tqdm(range(n_participants)):
    #         fit_participant = MAP(data[i], model_name='basic', iters=5,
    #                               only_mle=True)
    #         fit_participants.append(fit_participant)

    # print(fit_participants)
    # np.save("recovery_individual_mle.npy", fit_participants, allow_pickle=True)

    # %% run MLE for full data
    fit_pop_mle = MAP(data, model_name='basic', iters=40, only_mle=True)
    print(fit_pop_mle)
    np.save("recovery_group_mle.npy", fit_pop_mle, allow_pickle=True)

# %%
