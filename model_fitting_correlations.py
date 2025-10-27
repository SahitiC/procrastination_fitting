# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns
from scipy.stats import spearmanr

# %%


def get_correlation(a, b):
    mask = np.isnan(a)
    a = a[~mask]
    b = b[~mask]
    print(spearmanr(a, b))
    plt.scatter(a, b)


# %% import data
data_relevant = pd.read_csv('data_preprocessed.csv', index_col=False)

data_full = pd.read_csv('zhang_ma_data.csv', index_col=False)

result_fit = np.load("result_fit.npy", allow_pickle=True)

# %%
data_full_filter = data_full[data_full['SUB_INDEX_194'].isin(
    data_relevant['SUB_INDEX_194'])]
result_fit_params = np.vstack(np.hstack(result_fit[:, 1, :]))

discount_factors_log_empirical = np.array(data_full_filter['DiscountRate_lnk'])
discount_factors_fitted = result_fit_params[:, 0]
efficacy_fitted = result_fit_params[:, 1]
efforts_fitted = result_fit_params[:, 2]
proc_mean = np.array(data_full_filter['AcadeProcFreq_mean'])
impulsivity_score = np.array(data_full_filter['ImpulsivityScore'])
time_management = np.array(data_full_filter['ReasonProc_TimeManagement'])
task_aversiveness = np.array(data_full_filter['ReasonProc_TaskAversiveness'])

get_correlation(discount_factors_log_empirical, discount_factors_fitted)
get_correlation(proc_mean, discount_factors_fitted)
get_correlation(impulsivity_score, discount_factors_fitted)
get_correlation(task_aversiveness, np.abs(efforts_fitted))
get_correlation(time_management, efficacy_fitted)
