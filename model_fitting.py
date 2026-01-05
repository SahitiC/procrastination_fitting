# %% imports
import numpy as np
import pandas as pd
import ast
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import mle
import pickle
# import mle2
# import empirical_bayes

# %% functions


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

    # import data
    data_relevant = pd.read_csv(
        'data_preprocessed.csv', index_col=False)

    # convert into list from strings
    # multiply by two to convert hours to units
    units = list(data_relevant.apply(convert_string_to_list, axis=1))

    data_to_fit_lst = []
    for i in range(len(units)):
        data_to_fit_lst.append(np.array(units[i], dtype=int))

    # # save for matlab fitting
    # data = []
    # for i in range(len(units)):
    #     data.append([np.array(units[i], dtype=int)])

    # with open("data.pkl", "wb") as f:
    #     pickle.dump(data, f, protocol=5)

    def fit_single_mle(datum):
        return mle.MLE([[datum]], model_name='basic', iters=50)
        # return empirical_bayes.MAP([[datum]], model_name='basic', pop_means=np.array([0, 0, 0]),
        # pop_vars=np.array([6.25, 6.25, 1]), iters=15, only_mle=False, initial_guess=None)

    with ProcessPoolExecutor() as executor:
        fit_participants = list(tqdm(
            executor.map(fit_single_mle, data_to_fit_lst)))

    np.save("fit_individual_mle.npy", fit_participants, allow_pickle=True)

    data_to_fit_lst = np.array(data_to_fit_lst, dtype=object)
    np.save('data_to_fit_lst.npy', data_to_fit_lst)

    # %%
    # # fitting for clustered data
    # data_clustered = pd.read_csv(
    #     'data_clustered.csv', index_col=False)

    # units = list(data_clustered.apply(convert_string_to_list, axis=1))

    # data_to_fit_lst = []
    # for label in (np.unique(data_clustered['labels'])):
    #     data_cluster = []
    #     for i in range(len(units)):
    #         if data_clustered['labels'][i] == label:
    #             data_cluster.append([np.array(units[i], dtype=int)])
    #     data_to_fit_lst.append(data_cluster)

    # fit_pop_result = []
    # for i in range(len(data_to_fit_lst)):
    #     fit_pop = empirical_bayes.em(data_to_fit_lst[i], model_name='basic',
    #                                  max_iter=50, tol=1e-3,
    #                                  parallelise=True)
    #     fit_pop_result.append(fit_pop)

    # np.save("fit_pop_clusters.npy", fit_pop_result, allow_pickle=True)

    # data_to_fit_lst = np.array(data_to_fit_lst, dtype=object)
    # np.save('data_to_fit_lst_cluster.npy', data_to_fit_lst)
