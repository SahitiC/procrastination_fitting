# %%
import mle
import numpy as np
import cProfile
import pstats


def fit_single_mle():
    return mle.MLE([data], model_name='basic', iters=15)


data = [np.array([0,  1,  2,  5,  5,  8,  9, 10, 12,
                 12, 12, 12, 12, 12, 12, 14, 19])]
# %%

cProfile.run("fit_single_mle()", "fit_stats")
p = pstats.Stats('fit_stats')
p.sort_stats("cumulative").print_stats()

# %%
