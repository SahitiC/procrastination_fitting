"""
fit models to data, cluster-wise
"""

import numpy as np
import likelihoods
import concurrent.futures
import constants
import pandas as pd
import ast
import random

# %% functions


def model_fit(data_to_fit):
    """
    fit each model to inputted data using maximum likelihood estimation
    store likelihood and fitted parameters
    """

    # fit each model to data and recover params
    result_basic = likelihoods.maximum_likelihood_estimate_basic(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, constants.THR, constants.STATES_NO, data_to_fit)

    result_efficacy_gap = likelihoods.maximum_likelihood_estimate_efficacy_gap(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, constants.THR, constants.STATES_NO, data_to_fit)

    result_conv_conc = likelihoods.maximum_likelihood_estimate_convex_concave(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, constants.THR, constants.STATES_NO, data_to_fit)

    result_imm_basic = likelihoods.maximum_likelihood_estimate_immediate_basic(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, constants.THR, constants.STATES_NO, data_to_fit)

    result_diff_disc = likelihoods.maximum_likelihood_estimate_diff_discounts(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR_DIFF_DISCOUNTS,
        constants.REWARD_EXTRA_DIFF_DISCOUNTS, constants.REWARD_SHIRK,
        constants.BETA_DIFF_DISCOUNTS, constants.THR, constants.STATES_NO,
        data_to_fit)

    result_no_commit = likelihoods.maximum_likelihood_estimate_no_commitment(
        constants.STATES_NO_COMMIT, constants.INTEREST_STATES,
        constants.ACTIONS_BASE, constants.HORIZON, constants.REWARD_THR,
        constants.REWARD_EXTRA, constants.REWARD_SHIRK, constants.BETA,
        constants.P_STAY_LOW, constants.P_STAY_HIGH, constants.THR,
        constants.STATES_NO_NO_COMMIT, data_to_fit)

    # nllkhds under each model
    nllkhds = np.array([result_basic.fun, result_efficacy_gap.fun,
                        result_conv_conc.fun, result_imm_basic.fun,
                        result_diff_disc.fun, result_no_commit.fun])
    params = [result_basic.x, result_efficacy_gap.x,
              result_conv_conc.x, result_imm_basic.x,
              result_diff_disc.x, result_no_commit.x]

    return [nllkhds, params]


def convert_string_to_list(row):
    """
    convert into list from strings
    multiply by two to convert hours to units 
    """
    return np.array([0]+ast.literal_eval(
        row['cumulative progress weeks'])) * 2

# %% fit models

if __name__ == "__main__":

    np.random.seed(0)

    # import clustered data
    data_relevant = pd.read_csv(
        'data/data_clustered.csv', index_col=False)

    # convert into list from strings
    # multiply by two to convert hours to units        
    units = list(data_relevant.apply(convert_string_to_list, axis=1))

    # list of trajectory sets corresponding to each cluster
    # each model is fit to each of these clusters
    data_to_fit_lst = []
    for label in (np.unique(data_relevant['labels'])):
        data_cluster = []
        for i in range(len(units)):
            # only consider trajectories where max 22 units were completed
            if units[i][-1] <= 22:
                if data_relevant['labels'][i] == label:
                    data_cluster.append(np.array(units[i], dtype=int))
        data_to_fit_lst.append(data_cluster)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_lst = executor.map(model_fit, data_to_fit_lst)

    result_lst = [*result_lst]
    result = np.array(result_lst, dtype=object)
    np.save('data/result_fit.npy', result)

    data_to_fit_lst = np.array(data_to_fit_lst, dtype=object)
    np.save('data/data_to_fit_lst.npy', data_to_fit_lst)
