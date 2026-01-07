# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from scipy.optimize import minimize
# %% data

result_fit_mle = np.load(
    "fits/fit_individual_mle_with_hess.npy", allow_pickle=True)

data_relevant = pd.read_csv('data_preprocessed.csv', index_col=False)

result_fit_params = np.array([result_fit_mle[i]['par_b']
                              for i in range(len(result_fit_mle))])

result_fit_hess = np.array([result_fit_mle[i]['hess_b']
                            for i in range(len(result_fit_mle))])

# visualise fitted parameter distributions
for i in range(3):
    for j in range(i+1):
        plt.figure(figsize=(4, 4))
        plt.scatter(result_fit_params[:, i],
                    result_fit_params[:, j])
        plt.title(f'Param {i} vs Param {j}')

plt.figure(figsize=(4, 4))
plt.hist(np.log(1/result_fit_params[:, 0]))
plt.figure(figsize=(4, 4))
a = result_fit_params[:, 0]
a = np.where(a == 1, 0.99, a)
plt.hist(1/(1-a))
