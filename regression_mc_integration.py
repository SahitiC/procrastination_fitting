# %%
import numpy as np
import pandas as pd
import regression
import matplotlib.pyplot as plt
import gen_data
import constants
from scipy.special import logsumexp
from scipy.optimize import minimize
import statsmodels.formula.api as smf
from scipy.stats import chi2

# %%


def log_gaussian(y, mean, sigma):
    return (
        - np.log(sigma)
        - 0.5 * ((y - mean) / sigma)**2)


def log_gaussian_vec(x, xhat, sigma_x):
    return (
        - np.log(np.prod(sigma_x))
        - 0.5 * np.sum(((xhat - x)/sigma_x)**2, axis=1))


def log_likelihood_i_mc(pars, y_i, xhat_i, sigma_x_i, bounds,
                        x_samples):
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

    log_pxhat = log_gaussian_vec(x_samples, xhat_i, sigma_x_i)
    mean = x_samples @ beta + intercept
    log_py = log_gaussian(y_i, mean, sigma)

    return logsumexp(log_pxhat + log_py) - np.log(len(x_samples))


def negative_log_likelihood(pars, y, xhat, sigma_x, bounds,
                            x_samples):
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
            x_samples)
    return nll


def fit_regression(y, xhat, sigma_x, bounds, x_samples,
                   opt_bounds, initial_guess):

    result = minimize(negative_log_likelihood, initial_guess,
                      args=(y, xhat, sigma_x, bounds, x_samples),
                      bounds=opt_bounds)
    return result


def fit_restricted_regression(y, xhat, sigma_x, bounds, x_samples,
                              opt_bounds, initial_guess, restricted_indices):

    def negative_log_likelihood_null(pars, y, xhat, sigma_x, bounds,
                                     restricted_indices):

        p = xhat.shape[1]
        beta = np.zeros(p)
        free_idx = ~np.isin(np.arange(p), restricted_indices)
        beta[free_idx] = pars[0:free_idx.sum()]
        intercept = pars[free_idx.sum()]
        sigma = pars[free_idx.sum()+1]

        pars_restricted = np.r_[beta, intercept, sigma]
        nll = negative_log_likelihood(
            pars_restricted, y, xhat, sigma_x, bounds, x_samples)

        return nll

    result = minimize(negative_log_likelihood_null, initial_guess,
                      args=(y, xhat, sigma_x, bounds, restricted_indices),
                      bounds=opt_bounds)
    return result


def get_mucw_simulated(trajectory):
    if np.max(trajectory) > 14:
        a = np.where(trajectory >= 14)[0][0]
        arr = trajectory[:a+1]
        if arr[-1] > 14:
            arr[-1] = 14
        mucw = (14*(len(arr)+1) - np.sum(arr))/14
        return mucw
    else:
        arr = trajectory
        mucw = (np.max(arr)*(len(arr)+1) - np.sum(arr))/np.max(arr)
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

# %%

mucw = np.array(data_weeks.apply(regression.get_mucw, axis=1))
proc_mean = np.array(data['AcadeProcFreq_mean'])
discount_factors_log_empirical = np.array(data['DiscountRate_lnk'])
discount_factors_empirical = np.exp(discount_factors_log_empirical)

discount_factors_fitted = fit_params[:, 0]
efficacy_fitted = fit_params[:, 1]
efforts_fitted = fit_params[:, 2]

# %% test MC procedure with single regressor, compare to quad results
y, xhat, hess = regression.drop_nans(
    discount_factors_empirical, discount_factors_fitted, diag_hess[:, 0])

xhat_reshaped = xhat.reshape(-1, 1)

bounds = [(0, 1)]
x_samples = np.column_stack([np.random.uniform(a, b, size=5000)
                             for (a, b) in bounds])

result = fit_regression(y, xhat_reshaped,
                        (1/hess)**0.5,
                        bounds=[(0, 1)],
                        x_samples=x_samples,
                        opt_bounds=[
                            (None, None), (None, None), (1e-3, None)],
                        initial_guess=[1, 1, 1])

# %% regressions
np.random.seed(0)

y, x1, x2, x3, hess = regression.drop_nans(proc_mean, discount_factors_fitted,
                                           efficacy_fitted, efforts_fitted,
                                           diag_hess)

xhat = np.column_stack((x1, x2, x3))

bounds = [(0, 1), (0, 1), (-7, 0)]
x_samples = np.column_stack([np.random.uniform(a, b, size=5000)
                             for (a, b) in bounds])

result = fit_regression(
    y, xhat, (1/hess)**0.5,
    bounds=[(0, 1), (0, 1), (-7, 0)],
    x_samples=x_samples,
    opt_bounds=[(None, None), (None, None), (None, None),
                (None, None), (1e-3, None)],
    initial_guess=[1, 1, 1, 0.1, 1])

# null models
results_null = []
for i in range(3):
    results_null.append(fit_restricted_regression(
        y, xhat, (1/hess)**0.5,
        bounds=[(0, 1), (0, 1), (-7, 0)],
        x_samples=x_samples,
        opt_bounds=[(None, None), (None, None),
                    (None, None), (1e-3, None)],
        initial_guess=[1, 1, 0.1, 1],
        restricted_indices=[i]))

result_disc_only = fit_restricted_regression(
    y, xhat, (1/hess)**0.5,
    bounds=[(0, 1), (0, 1), (-7, 0)],
    x_samples=x_samples,
    opt_bounds=[(None, None), (None, None), (1e-3, None)],
    initial_guess=[1, 0.1, 1],
    restricted_indices=[1, 2])


# %%
for i in range(3):
    lr_stat = 2 * (results_null[i].fun - result.fun)
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    print(lr_stat, p_value)

lr_stat = 2 * (result_disc_only.fun - result.fun)
p_value = 1 - chi2.cdf(lr_stat, df=2)
print(lr_stat, p_value)

# %% ols regression
df = pd.DataFrame({'y': y,
                   'disc': x1,
                   'efficacy': x2,
                   'effort': x3})
model1 = smf.ols(
    formula='y ~ disc + efficacy + effort', data=df).fit()
print(model1.summary())

model0 = smf.ols(
    formula='y ~ disc', data=df).fit()
print(model0.summary())

lr_stat, p_value, df_diff = model1.compare_lr_test(model0)
print(lr_stat, p_value, df_diff)

# %% plots

y, disc, efficacy, effort = regression.drop_nans(
    mucw, discount_factors_fitted, efficacy_fitted,
    efforts_fitted)
plt.figure(figsize=(4, 4))
plt.scatter(disc, y)
plt.xlabel('discount factor')
plt.figure(figsize=(4, 4))
plt.scatter(efficacy, y)
plt.xlabel('efficacy')
plt.figure(figsize=(4, 4))
plt.scatter(effort, y)
plt.xlabel('effort')

a = disc
a = np.where(disc == 1, 0.99, disc)
plt.figure(figsize=(4, 4))
plt.scatter(1/(1-a), y)
plt.xlabel('1/(1-disc)')

#  compare with simulated data for these parameters
mucw_simulated = []
for i in range(len(disc)):
    data = gen_data.gen_data_basic(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA,
        constants.REWARD_SHIRK, constants.BETA, disc[i], efficacy[i],
        effort[i], 5, constants.THR, constants.STATES_NO)
    temp = []
    for d in data:
        temp.append(get_mucw_simulated(d))
    mucw_i = np.nanmean(np.array(temp))
    mucw_simulated.append(mucw_i)
plt.figure(figsize=(4, 4))
plt.scatter(disc, mucw_simulated)
plt.xlabel('discount factor')
plt.figure(figsize=(4, 4))
plt.scatter(1/(1-a), mucw_simulated)
plt.xlabel('1/(1-disc)')
plt.figure(figsize=(4, 4))
plt.scatter(efficacy, mucw_simulated)
plt.xlabel('eficacy')
plt.figure(figsize=(4, 4))
plt.scatter(effort, mucw_simulated)
plt.xlabel('effort')

# %%
