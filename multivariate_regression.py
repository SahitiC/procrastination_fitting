# %%
import numpy as np
import pandas as pd
import ast
import regression

# %%


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

# %%


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

valid_indices = np.where(np.all(result_diag_hess > 0, axis=1))[0]

data = data_full_filter.iloc[valid_indices].reset_index(drop=True)
data_weeks = data_relevant.iloc[valid_indices].reset_index(drop=True)
fit_params = result_fit_params[valid_indices]
diag_hess = result_diag_hess[valid_indices]

mucw = np.array(data_weeks.apply(get_mucw, axis=1))
proc_mean = np.array(data['AcadeProcFreq_mean'])

discount_factors_fitted = fit_params[:, 0]
efficacy_fitted = fit_params[:, 1]
efforts_fitted = fit_params[:, 2]

# %% regressions

y, x1, x2, x3, hess = drop_nans(mucw, discount_factors_fitted,
                                efficacy_fitted, efforts_fitted, diag_hess)

xhat = np.column_stack((x1, x2, x3))

result = regression.fit_regression(
    y, xhat, (1/hess)**0.5,
    bounds=[(0, 1), (0, 1), (-7, 0)],
    opt_bounds=[(None, None), (None, None), (None, None),
                (None, None), (1e-3, None)],
    initial_guess=[1, 1, 1, 0.1, 1])

np.save('result_regression.npy', result)
