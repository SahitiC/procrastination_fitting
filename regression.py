# %% imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.integrate import nquad
import matplotlib.pyplot as plt
import ast

# %% functions


def integrand(*args):
    """
    Multivariate normal integrand for regression with measurement error.
    Calculates integrand for each observation (y_i, x_hat_i) and some values of
    the latent predictors (x1, x2, x3).

    Parameters
    ----------
    args = (x1, x2, ..., xp, y_i, xhat_i, beta, intercept, sigma, sigma_x_i)
    x1, x2, ..., xp : float
        Variables of integration. Latent predictors in the regression model.
    y_i : float
        Observed dependent variable.
    xhat_i : array-like, shape (p,)           
        Observed estimates of the predictors with measurement error.
    beta : array-like, shape (p,)
        Regression coefficients.
    intercept : float
        Regression intercept.
    sigma : float
        Standard deviation of the measurement error in y.
    sigma_x_i : array-like, shape (p,)
        Standard deviations of the measurement errors in predictors. 
        Given by inverse hessian.
    """
    *xs, y_i, xhat_i, sigma_x_i, beta, intercept, sigma = args
    x = np.array(xs)
    integrand = (1/(np.prod(sigma_x_i)*sigma) *
                 np.exp(-0.5 * np.sum(((x - xhat_i)/sigma_x_i)**2) +
                        -0.5 * ((y_i - (np.dot(beta, x) + intercept))/sigma)**2))
    return integrand


def likelihood_i(pars, y_i, xhat_i, sigma_x_i, bounds):
    """
    Likelihood for a single observation (y_i, x_hat_i) given parameters.

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
    """
    p = len(xhat_i)  # no. of predictors
    beta = pars[0:p]
    intercept = pars[p]
    sigma = pars[p+1]

    integral, error = nquad(integrand, bounds,
                            args=(y_i, xhat_i, sigma_x_i, beta, intercept, sigma))

    return integral


def negative_log_likelihood(pars, y, xhat, sigma_x, bounds):
    """
    Negative log likelihood for the entire dataset.

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
    """
    nll = 0

    for i in range(len(y)):
        ll_i = likelihood_i(pars, y[i], xhat[i], sigma_x[i], bounds)
        nll -= np.log(ll_i + 1e-10)  # add small constant to avoid log(0)

    return nll


def fit_regression(y, xhat, sigma_x, bounds, ranges, initial_guess):
    """
    Fit regression model using MLE.

    Parameters
    ----------
    y : array-like, shape (n_observations,)
        Observed dependent variable.
    xhat : array-like, shape (n_observations, p)
        Observed estimates of the predictors with measurement error.
    sigma_x : array-like, shape (n_observations, p)
        Standard deviations of the measurement errors in predictors.
    initial_guess : array-like, shape (p+2,), optional
        Initial guess for the parameters: [beta_1, beta_2, ..., beta_p, intercept, sigma]
    bounds : list of tuples
        Integration bounds for each predictor.  
    ranges : list of tuples
        Bounds for the optimization parameters.

    Returns
    -------
    result : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
    """

    result = minimize(negative_log_likelihood, initial_guess,
                      args=(y, xhat, sigma_x, bounds),
                      bounds=ranges)
    return result


def regression_model(y, xhat, sigma_x):
    pass


def drop_nans(*arrays):
    stacked = np.column_stack(arrays)
    mask = np.isnan(stacked).any(axis=1)
    return tuple(arr[~mask] for arr in arrays)


def get_mucw(row):
    units = np.array(ast.literal_eval(
        row['delta progress weeks']))*2
    units_cum = np.array(ast.literal_eval(
        row['cumulative progress weeks']))*2
    if np.max(units_cum) >= 14:
        a = np.where(units_cum >= 14)[0][0]
        arr = units[:a+1]
        mucw = np.sum(arr * np.arange(1, len(arr)+1)) / 14
        return mucw
    else:
        arr = units
        mucw = np.sum(arr * np.arange(1, len(arr)+1)) / np.sum(arr)
        return mucw

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
# plot params

for i in range(3):
    plt.figure(figsize=(4, 4))
    plt.hist(fit_params[:, i])

for i in range(3):
    for j in range(i+1):
        plt.figure(figsize=(4, 4))
        plt.scatter(fit_params[:, i],
                    fit_params[:, j])
        plt.title(f'Param {i} vs Param {j}')

plt.figure(figsize=(4, 4))
plt.hist(np.log(1/fit_params[:, 0]))
plt.figure(figsize=(4, 4))
a = fit_params[:, 0]
a = np.where(a == 1, 0.99, a)
plt.hist(1/(1-a))

# %% variables

discount_factors_log_empirical = np.array(data['DiscountRate_lnk'])
discount_factors_empirical = np.exp(discount_factors_log_empirical)
proc_mean = np.array(data['AcadeProcFreq_mean'])
impulsivity_score = np.array(data['ImpulsivityScore'])

discount_factors_fitted = fit_params[:, 0]
efficacy_fitted = fit_params[:, 1]
efforts_fitted = fit_params[:, 2]

mucw = np.array(data_weeks.apply(get_mucw, axis=1))

# %% regressions

y, xhat = drop_nans(proc_mean, discount_factors_fitted)

xhat_reshaped = xhat.reshape(-1, 1)

fit_regression(y, xhat_reshaped,
               (1/diag_hess[:, 0])**0.5,
               bounds=[(0, 1)],
               ranges=[(None, None), (None, None), (1e-3, None)],
               initial_guess=[0.1, 0.1, 1])

# %%

y, x1, x2, x3 = drop_nans(mucw, discount_factors_fitted,
                          efficacy_fitted, efforts_fitted)

xhat = np.column_stack((x1, x2, x3))

fit_regression(y, xhat,
               (1/diag_hess)**0.5,
               bounds=[(0, 1), (0, 1), (-7, 0)],
               ranges=[(None, None), (None, None), (None, None), (None, None),
                       (1e-3, None)],
               initial_guess=[0.1, 0.1, 0.1, 0.1, 1])
