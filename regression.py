# %% imports
from sklearn.cross_decomposition import CCA
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import nquad
import matplotlib.pyplot as plt
import ast
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from scipy.stats import chi2
from statsmodels.stats.mediation import Mediation
import statsmodels.formula.api as smf

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
    # uniform = 1 / np.prod([b - a for a, b in bounds])
    integrand = (
                (1/(np.prod(sigma_x_i)*sigma)) *
        np.exp(
                    -0.5 * np.sum(((xhat_i - x)/sigma_x_i)**2) +
                    -0.5 * ((y_i - (np.dot(beta, x) + intercept))/sigma)**2))
    return integrand


def likelihood_i(pars, y_i, xhat_i, sigma_x_i, bounds):
    """
    Likelihood for a single observation (y_i, x_hat_i) given parameters. Avoid
    for more than one predictor x_i due to compuational cost of integration.

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

    Returns
    -------
    float
        Likelihood for individual i.
    """
    p = len(xhat_i)  # no. of predictors
    beta = pars[0:p]
    intercept = pars[p]
    sigma = pars[p+1]

    integral, error = nquad(
        integrand, bounds,
        args=(y_i, xhat_i, sigma_x_i, beta, intercept, sigma))

    return integral


def negative_log_likelihood(pars, y, xhat, sigma_x, bounds):
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

    Returns
    -------
    float
        Negative log-likelihood for the dataset.
    """

    nll = 0
    for i in range(len(y)):
        ll_i = likelihood_i(pars, y[i], xhat[i], sigma_x[i], bounds)
        nll -= np.log(ll_i + 1e-10)  # add small constant to avoid log(0)
    return nll


def fit_regression(y, xhat, sigma_x, bounds, opt_bounds, initial_guess):
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
        Initial guess for the parameters: [beta_1, beta_2, ..., beta_p,
        intercept, sigma]
    bounds : list of tuples
        Integration bounds for each predictor.
    opt_bounds : list of tuples
        Bounds for the optimization parameters.

    Returns
    -------
    result : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
    """
    result = minimize(negative_log_likelihood, initial_guess,
                      args=(y, xhat, sigma_x, bounds),
                      bounds=opt_bounds)
    return result


def fit_null_regression(y, xhat, sigma_x, bounds, opt_bounds, initial_guess):
    """
    Fit null regression model (intercept only) using MLE.

    Parameters
    ----------
    y : array-like, shape (n_observations,)
        Observed dependent variable.
    opt_bounds : list of tuples
        Bounds for the optimization parameters.
    initial_guess : array-like, shape (2,), optional
        Initial guess for the parameters: [intercept, sigma]
    Returns
    -------
    result : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
    """

    def negative_log_likelihood_null(pars, y, xhat, sigma_x, bounds):
        # pars = [intercept, sigma]
        p = xhat.shape[1]
        beta = np.zeros(p)
        intercept, sigma = pars

        nll = 0.0
        for i in range(len(y)):
            ll_i = likelihood_i(
                np.r_[beta, intercept, sigma],
                y[i], xhat[i], sigma_x[i], bounds)
            nll -= np.log(ll_i + 1e-10)
        return nll

    result = minimize(negative_log_likelihood_null, initial_guess,
                      args=(y, xhat, sigma_x, bounds),
                      bounds=opt_bounds)
    return result


def drop_nans(*arrays):
    stacked = np.column_stack(arrays)
    mask = np.isnan(stacked).any(axis=1)
    return tuple(arr[~mask] for arr in arrays)


def get_mucd(row):
    units = np.array(ast.literal_eval(
        row['delta progress']))*2
    units_cum = np.array(ast.literal_eval(
        row['cumulative progress']))*2
    if np.max(units_cum) > 14:
        a = np.where(units_cum >= 14)[0][0]
        arr = units[:a+1]
        if units_cum[a] > 14:
            arr[-1] = 14 - units_cum[a-1]
        mucd = np.sum(arr * np.arange(1, len(arr)+1)) / 14
        return mucd
    else:
        arr = units
        mucd = np.sum(arr * np.arange(1, len(arr)+1)) / np.sum(arr)
        return mucd


def get_mucw(row):
    units = np.array(ast.literal_eval(
        row['delta progress weeks']))*2
    units_cum = np.array(ast.literal_eval(
        row['cumulative progress weeks']))*2
    if np.max(units_cum) > 14:
        a = np.where(units_cum >= 14)[0][0]
        arr = units[:a+1]
        if units_cum[a] > 14:
            arr[-1] = 14 - units_cum[a-1]
        mucw = np.sum(arr * np.arange(1, len(arr)+1)) / 14
        return mucw
    else:
        arr = units
        mucw = np.sum(arr * np.arange(1, len(arr)+1)) / np.sum(arr)
        return mucw


def get_completion_week(row):
    hours = np.array(ast.literal_eval(
        row['cumulative progress weeks']))
    if np.max(hours) >= 7:
        return np.where(hours >= 7)[0][0]
    else:
        return np.nan


# %%

if __name__ == "__main__":

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

    # %% basic correlations

    discount_factors_log_empirical = np.array(
        data_full_filter['DiscountRate_lnk'])
    discount_factors_empirical = np.exp(discount_factors_log_empirical)
    impulsivity_score = np.array(data_full_filter['ImpulsivityScore'])
    self_control = np.array(data_full_filter['SelfControlScore'])
    proc_mean = np.array(data_full_filter['AcadeProcFreq_mean'])
    mucw = np.array(data_relevant.apply(get_mucw, axis=1))
    mucd = np.array(data_relevant.apply(get_mucd, axis=1))
    completion_week = np.array(
        data_relevant.apply(get_completion_week, axis=1))

    y, x = drop_nans(completion_week, discount_factors_empirical)
    pearsonr(y, x)

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
    a = fit_params[:, 0]
    a = np.where(a == 1, 0.999, a)
    plt.hist(1/(1-a))

    # %% variables

    # survey results
    discount_factors_log_empirical = np.array(data['DiscountRate_lnk'])
    discount_factors_empirical = np.exp(discount_factors_log_empirical)
    proc_mean = np.array(data['AcadeProcFreq_mean'])
    impulsivity_score = np.array(data['ImpulsivityScore'])
    time_management = np.array(data['ReasonProc_TimeManagement'])
    task_aversiveness = np.array(data['ReasonProc_TaskAversiveness'])
    laziness = np.array(data['ReasonProc_Laziness'])
    self_control = np.array(data['SelfControlScore'])

    # fitted parameters
    discount_factors_fitted = fit_params[:, 0]
    efficacy_fitted = fit_params[:, 1]
    efforts_fitted = fit_params[:, 2]

    # delays
    mucw = np.array(data_weeks.apply(get_mucw, axis=1))
    completion_week = np.array(data_weeks.apply(get_completion_week, axis=1))

    # %% regressions

    y, xhat, hess = drop_nans(
        self_control, discount_factors_fitted, diag_hess[:, 0])

    xhat_reshaped = xhat.reshape(-1, 1)

    # error regression with one prdictor
    result = fit_regression(y, xhat_reshaped,
                            (1/hess)**0.5,
                            bounds=[(0, 1)],
                            opt_bounds=[
                                (None, None), (None, None), (1e-3, None)],
                            initial_guess=[0.1, 0.1, 1])
    print(result)

    # null regression model with only intercept (and sigma_y ofc)
    result_null = fit_null_regression(y, xhat_reshaped,
                                      (1/hess)**0.5,
                                      bounds=[(0, 1)],
                                      opt_bounds=[(None, None), (1e-3, None)],
                                      initial_guess=[0.1, 1])
    print(result_null)

    # LRT
    lr_stat = 2 * (result_null.fun - result.fun)
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    print(lr_stat, p_value)

    # corresponding OLS regressions
    df = pd.DataFrame({'y': y,
                       'xhat': xhat})
    model = smf.ols(
        formula='y ~ xhat', data=df).fit()

    df = pd.DataFrame({'y': y})
    model0 = smf.ols(
        formula='y ~ 1', data=df).fit()

    # %% why no effect of efficacy and effort on proc_mean
    # should we do a mediation analysis - i.e does PASS mediate effect of
    # disc on mucw

    Pass, Mucw, discount, efficacy, effort = drop_nans(
        proc_mean, mucw, discount_factors_fitted, efficacy_fitted,
        efforts_fitted)

    df = pd.DataFrame({'proc_mean': Pass,
                       'mucw': Mucw,
                       'discount': discount,
                       'efficacy': efficacy,
                       'effort': effort})
    model1 = smf.ols(formula='mucw ~ proc_mean', data=df).fit()
    model2 = smf.ols(
        formula='mucw ~ discount + efficacy + effort', data=df).fit()
    model3 = smf.ols(
        formula='mucw ~ discount + efficacy + effort + proc_mean',
        data=df).fit()
    model4 = smf.ols(
        formula='mucw ~ proc_mean + efficacy + effort', data=df).fit()

    # %%

    df = pd.DataFrame({'pass': proc_mean,
                       'disc_emp': discount_factors_empirical,
                       'impulsivity': impulsivity_score,
                       'self_control': self_control,
                       'time_man': time_management,
                       'task_avers': task_aversiveness,
                       'disc': discount_factors_fitted,
                       'efficacy': efficacy_fitted,
                       'effort': efforts_fitted})

    df = df.dropna()
    df = (df-df.mean())/df.std()

    X = df.iloc[:, 0:6]
    Y = df.iloc[:, 6:9]

    cca = CCA(n_components=2)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    score = cca.score(X, Y)

    plt.figure()
    plt.scatter(X_c[:, 0], Y_c[:, 0])
    print(pearsonr(X_c[:, 0], Y_c[:, 0]))
    plt.figure()
    plt.scatter(X_c[:, 1], Y_c[:, 1])
    print(pearsonr(X_c[:, 1], Y_c[:, 1]))

    print(cca.x_loadings_)
    print(cca.y_loadings_)

    print(cca.x_weights_)
    print(cca.y_weights_)

    # %% mediation analysis

    y, m, disc, effc, efft = drop_nans(
        mucw, proc_mean, discount_factors_fitted, efficacy_fitted,
        efforts_fitted)

    df = pd.DataFrame({'y': y,
                       'm': m,
                       'disc': disc,
                       'effc': effc,
                       'efft': efft})

    model1 = smf.ols(formula='m ~ disc + effc + efft', data=df)
    model2 = smf.ols(formula='y ~ m + disc + effc + efft', data=df)

    med = Mediation(model2, model1, exposure='disc', mediator='m')
    med_result = med.fit(n_rep=5000, method='bootstrap')
    print(med_result.summary())

    # %%
    y, m, disc = drop_nans(
        mucw, proc_mean, discount_factors_empirical)

    df = pd.DataFrame({'y': y,
                       'm': m,
                       'disc': disc})

    model1 = smf.ols(formula='m ~ disc', data=df)
    model2 = smf.ols(formula='y ~ m + disc', data=df)

    med = Mediation(model2, model1, exposure='disc', mediator='m')
    med_result = med.fit(n_rep=5000, method='bootstrap')
    print(med_result.summary())
