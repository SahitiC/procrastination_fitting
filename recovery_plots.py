# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
input_data_recovery = np.load("input_data_recovery.npy", allow_pickle=True)
recovery_em = np.load("recovery_em.npy", allow_pickle=True)
recovery_individual_mle = np.load("recovery_individual_mle.npy",
                                  allow_pickle=True)
recovery_group_mle = np.load("recovery_group_mle.npy", allow_pickle=True)

# %%
