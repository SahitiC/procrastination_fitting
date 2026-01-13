# %%
import numpy as np
import pandas as pd
import regression
from scipy.special import logsumexp
from scipy.optimize import minimize

# %%


def log_gaussian(y, mean, sigma):
    return (
        - np.log(sigma)
        - 0.5 * ((y - mean) / sigma)**2)


def log_gaussian_vec(x, xhat, sigma_x):
    return (
        - np.log(np.prod(sigma_x))
        - 0.5 * np.sum(((xhat - x)/sigma_x)**2))


def log_likelihood_i_mc(pars, y_i, xhat_i, sigma_x_i, bounds,
                        sample_size_mc=1000):
    """
    Likelihood for a single observation (y_i, x_hat_i) given parameters using
    Monte Carlo integration.

    Parameters
    ----------
    pars : array-like, shape (p+2,)
        Model parameters: [beta_1, beta_2, ..., beta_p, intercept, sigma]
    y_i : float
        Observed dependent variable.
    xhat_i : array-like, shape (p,)
        Observed estimates of the predictors with measurement error.
    sigma_x_i : array-like, shape (p,)
        Standard deviations of the measurement errors in predictors.
    bounds : list of tuples
        Integration bounds for each predictor.
    sample_size : int, optional
        Number of Monte Carlo samples to draw. Default is 1000.

    Returns
    -------
    float
        Log-likelihood for individual i.
    """

    p = len(xhat_i)  # no. of predictors
    beta = pars[0:p]
    intercept = pars[p]
    sigma = pars[p+1]

    x_samples = np.column_stack([
        np.random.uniform(a, b, size=sample_size_mc)
        for (a, b) in bounds])

    print(x_samples)
    print(xhat_i)
    log_pxhat = log_gaussian_vec(x_samples, xhat_i, sigma_x_i).sum(axis=1)
    mean = x_samples @ beta + intercept
    log_py = log_gaussian(y_i, mean, sigma)

    return logsumexp(log_pxhat + log_py) - np.log(sample_size_mc)


def negative_log_likelihood(pars, y, xhat, sigma_x, bounds,
                            sample_size_mc=1000):
    """
    Negative log likelihood for the entire dataset. Avoid for more than one
    predictor x_i due to compuational cost of integration.

    Parameters
    ----------
    pars : array-like, shape (p+2,)
        Model parameters: [beta_1, beta_2, ..., beta_p, intercept, sigma]
    y : array-like, shape (n_observations,)
        Observed dependent variable.
    xhat : array-like, shape (n_observations, p)
        Observed estimates of the predictors with measurement error.
    sigma_x : array-like, shape (n_observations, p)
        Standard deviations of the measurement errors in predictors.
    bounds : list of tuples
        Integration bounds for each predictor.
    sample_size_mc : int, optional
        Number of Monte Carlo samples to draw. Default is 1000.

    Returns
    -------
    float
        Negative log-likelihood for the dataset.
    """

    nll = 0
    for i in range(len(y)):
        nll -= log_likelihood_i_mc(
            pars, y[i], xhat[i], sigma_x[i], bounds,
            sample_size_mc=sample_size_mc)
    return nll


def fit_regression(y, xhat, sigma_x, bounds, sample_size_mc,
                   opt_bounds, initial_guess):

    result = minimize(negative_log_likelihood, initial_guess,
                      args=(y, xhat, sigma_x, bounds, sample_size_mc),
                      bounds=opt_bounds)
    return result

# %% data


data_relevant = pd.read_csv('data_preprocessed.csv', index_col=False)

data_full = pd.read_csv('zhang_ma_data.csv',
                        index_col=False)

data_full_filter = data_full[data_full['SUB_INDEX_194'].isin(
    data_relevant['SUB_INDEX_194'])]
data_full_filter = data_full_filter.reset_index(drop=True)

result_fit_mle = np.load(
    "fits/fit_individual_mle_with_hess.npy", allow_pickle=True)

result_fit_params = np.array([result_fit_mle[i]['par_b']
                              for i in range(len(result_fit_mle))])

result_diag_hess = np.array([result_fit_mle[i]['hess_diag']
                            for i in range(len(result_fit_mle))])

# %% remove ppts with negative hess

valid_indices = np.where(np.all(result_diag_hess > 0, axis=1))[0]

data = data_full_filter.iloc[valid_indices].reset_index(drop=True)
data_weeks = data_relevant.iloc[valid_indices].reset_index(drop=True)
fit_params = result_fit_params[valid_indices]
diag_hess = result_diag_hess[valid_indices]

# %%

mucw = np.array(data_weeks.apply(regression.get_mucw, axis=1))
proc_mean = np.array(data['AcadeProcFreq_mean'])

discount_factors_fitted = fit_params[:, 0]
efficacy_fitted = fit_params[:, 1]
efforts_fitted = fit_params[:, 2]

# %% regressions

y, x1, x2, x3, hess = regression.drop_nans(mucw, discount_factors_fitted,
                                           efficacy_fitted, efforts_fitted,
                                           diag_hess)

xhat = np.column_stack((x1, x2, x3))

result = fit_regression(
    y, xhat, (1/hess)**0.5,
    bounds=[(0, 1), (0, 1), (-7, 0)],
    sample_size_mc=500,
    opt_bounds=[(None, None), (None, None), (None, None),
                (None, None), (1e-3, None)],
    initial_guess=[1, 1, 1, 0.1, 1])

# np.save('result_regression.npy', result)

# %%
