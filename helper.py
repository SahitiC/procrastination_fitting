# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sample_initial_params(model_name, num_samples=1):
    """Sample initial parameters for MAP estimation."""

    if model_name == 'rl-basic':
        alpha_u = np.random.logistic(0, 1)
        beta_u = np.random.logistic(0, 1)
        pars = [alpha_u, beta_u]

    elif model_name == 'basic':
        discount_factor = np.random.logistic(0, 1)
        efficacy = np.random.logistic(0, 1)
        effort_work = -1 * np.random.exponential(0.5)
        pars = [discount_factor, efficacy, effort_work]

    return pars


def sample_params(num_samples=1):
    """Sample parameters to generate data."""

    pars = []
    for _ in range(num_samples):
        discount_factor = np.random.uniform(0.2, 1)
        efficacy = np.random.uniform(0.35, 1)
        effort_work = -1 * np.random.exponential(0.5)
        pars.append([discount_factor, efficacy, effort_work])

    return pars


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


def plot_clustered_data(data, labels):
    """
    plot trajectories in each cluster given data and labels
    """

    for label in set(labels):
        plt.figure(figsize=(4, 4), dpi=300)

        for i, d in enumerate(data):
            if labels[i] == label:
                plt.plot(np.array(d) * 2, alpha=0.5)

        sns.despine()
        plt.xticks([0, 7, 15])
        plt.yticks([0, 11, 14, 22])
        plt.xlabel('time (weeks)')
        plt.ylabel('research units \n completed')
        plt.show()


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


# %%
