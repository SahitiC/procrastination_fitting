# %% imports
import numpy as np
import pandas as pd
import ast
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import empirical_bayes

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

    # import clustered data
    data_relevant = pd.read_csv(
        'data_preprocessed.csv', index_col=False)

    # convert into list from strings
    # multiply by two to convert hours to units
    units = list(data_relevant.apply(convert_string_to_list, axis=1))

    # list of trajectory sets corresponding to each cluster
    # each model is fit to each of these clusters
    data_to_fit_lst = []
    for i in range(len(units)):
        data_to_fit_lst.append(np.array(units[i], dtype=int))

    fit_pop_result = empirical_bayes.em(data_to_fit_lst, model_name='basic',
                                        max_iter=20, tol=1e-3,
                                        parallelise=True)

    fit_pop_mle = empirical_bayes.MAP(
        data_to_fit_lst, model_name='basic', iters=50, only_mle=True)

    def fit_single_mle(datum):
        return empirical_bayes.MAP(datum, model_name='basic',
                                   iters=20, only_mle=True)

    with ProcessPoolExecutor() as executor:
        fit_participants = list(tqdm(
            executor.map(fit_single_mle, data_to_fit_lst)))

    np.save("fit_pop_em.npy", fit_pop_result, allow_pickle=True)

    np.save("fit_individual_mle.npy", fit_participants, allow_pickle=True)

    np.save("fit_pop_mle.npy", fit_pop_mle, allow_pickle=True)

    data_to_fit_lst = np.array(data_to_fit_lst, dtype=object)
    np.save('data_to_fit_lst.npy', data_to_fit_lst)
