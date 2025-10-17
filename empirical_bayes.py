# %%
import numpy as np
from scipy.optimize import minimize
import likelihoods
import constants
import gen_data
from tqdm import tqdm

# %%


def get_num_params(model_name):
    model_params = {
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

    if model_name == 'basic':
        nllkhd = likelihoods.likelihood_basic_model(
            params, constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, constants.THR,
            constants.STATES_NO, data)

    return nllkhd


def sample_initial_params(model_name, num_samples=1):
    """Sample initial parameters for MAP estimation."""

    if model_name == 'basic':
        discount_factor = np.random.uniform(0, 1)
        efficacy = np.random.uniform(0, 1)
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
            # forward: bounded = high - exp(par)
            # inverse: par = log(high - bounded)  (bounded < high)
            unbounded_pars[i] = np.log(high - x)
        elif low == 0 and high is None:
            # forward: bounded = low + exp(par)
            # inverse: par = log(bounded - low)
            unbounded_pars[i] = np.log(x - low)
        else:
            # forward: bounded = low + (high - low)/(1 + exp(-par))
            # inverse: par = logit((bounded - low)/(high - low))
            ratio = (x - low) / (high - low)
            # clip to avoid log(0)
            ratio = np.clip(ratio, 1e-9, 1 - 1e-9)
            unbounded_pars[i] = np.log(ratio / (1 - ratio))
    return unbounded_pars


def MAP(data_participant, pop_means, pop_vars, model_name, iters=5):
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

        log_lik = compute_log_likelihood(pars, data_participant, model_name)
        pars_unbounded = trans_to_unbounded(pars, param_ranges)
        log_prior = - (len(pars) / 2.) * np.log(2 * np.pi) - np.sum(
            np.log(pop_vars)) / 2. - sum((pars_unbounded - pop_means) ** 2. / (
                2 * pop_vars))

        return (log_lik - log_prior)

    # optimization
    for iter in range(iters):
        post = np.inf
        pars = sample_initial_params(model_name)
        res = minimize(neg_log_post, pars, bounds=param_ranges)
        if res.fun < post:
            post = res.fun
            final_res = res

    # compute hessian at the optimum
    diag_hess = Hess_diag(neg_log_post, final_res.x)
    par_u = trans_to_unbounded(final_res.x, param_ranges)

    fit_participant = {'par_b': final_res.x,  # bounded params
                       'par_u': par_u,  # unbounded params
                       'diag_hess': diag_hess,
                       'neg_log_post': final_res.fun,
                       'success': final_res.success}

    return fit_participant


def em(data, num_participants, model_name, max_iter=100, tol=1e-6):
    """
    Run the EM algorithm for a given model and data.

    Parameters:
    - data: array-like, shape (n_participants,)
        The input data for the EM algorithm.
    - model_name: str
        The name of the model to use ('gaussian', 'bernoulli', etc.).
    - max_iter: int, optional (default=100)
        The maximum number of iterations to run the EM algorithm.
    - tol: float, optional (default=1e-6)
        The tolerance for convergence.

    Returns:
    - params: dict
        The estimated parameters of the model.
    """

    n_params = get_num_params(model_name)

    # initialise prior
    pop_means = np.random.randn(n_params)  # or np.zeros(n_params)
    pop_vars = np.ones(n_params) * 6.25

    # EM algorithm

    for iteration in tqdm(range(max_iter)):
        # E-step
        fit_participants = []

        for i in range(num_participants):

            fit_participant = MAP(data[i], pop_means, pop_vars, model_name)
            fit_participants.append(fit_participant)

        # M-step
        pars_U = np.array([fit_participant['par_u']
                           for fit_participant in fit_participants])
        diag_hess = np.array([fit_participant['diag_hess']
                              for fit_participant in fit_participants])

        new_pop_means = np.mean(pars_U, axis=0)
        new_pop_vars = np.mean(pars_U**2.+1./diag_hess,
                               axis=0)-new_pop_means**2.
        print(np.abs(new_pop_means-pop_means), np.abs(new_pop_vars-pop_vars))
        # check convergence
        if np.max(np.abs(new_pop_means-pop_means)) < tol and np.max(
                np.abs(new_pop_vars-pop_vars)) < tol:

            print(f'Converged in {iteration} iterations.')
            pop_means = new_pop_means
            pop_vars = new_pop_vars
            break

        pop_means = new_pop_means.copy()
        pop_vars = new_pop_vars.copy()

    # bic = compute_bic(data, fit_participants, pop_means,
    #                   pop_vars, model_name)

    fit_pop = {'pop_means': pop_means, 'pop_vars': pop_vars,
               'fit_participants': fit_participants, 'model_name': model_name}

    return fit_pop


# %%

data = gen_data.gen_data_basic(
    constants.STATES, constants.ACTIONS,  constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
    constants.BETA, 0.8, 0.9, -0.2, 5, constants.THR, constants.STATES_NO)

fit_pop = em(data, num_participants=5, model_name='basic', max_iter=20)

print(fit_pop)

# %%
