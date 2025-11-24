# %%

import numpy as np
import pandas as pd
import ast
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import helper

# %% functions


def normalize_cumulative_progress(row):
    """
    normalise cumulative progress by total credits
    """
    temp = np.array(ast.literal_eval(row['cumulative progress']))
    return list(temp / row['Total credits'])


def process_delta_progress(row, semester_length_weeks):
    """
    aggregate delta progress over days to weeks
    """
    temp = ast.literal_eval(row['delta progress'])
    temp_week = [sum(temp[i_week*7: (i_week+1)*7])
                 for i_week in range(semester_length_weeks)]

    assert sum(temp_week) == row['Total credits']
    return temp_week


def cumulative_progress_weeks(row):

    return list(np.cumsum(row['delta progress weeks']))


def get_timeseries_to_cluster(row):

    return row['cumulative progress weeks']


# %% drop unwanted rows

data = pd.read_csv('zhang_ma_data.csv')

# drop the ones that discontinued (subj. 1, 95, 111)
# they report to have discountinued in verbal response in 'way_allocate' column
# they do 1 hour in the very beginning and then nothing after
# sbj 24, 55, 126 also dont finish 7 hours but not because they drop out
data_relevant = data.drop([1, 95, 111])
data_relevant = data_relevant.reset_index(drop=True)

# drop NaN entries
data_relevant = data_relevant.dropna(subset=['delta progress'])
data_relevant = data_relevant.reset_index(drop=True)

# drop ones who complete more than 11 hours
# as extra credit ends at 11 hours
# we do not consider extra rewards for > 11 hours in our models as well
mask = np.where(data_relevant['Total credits'] <= 11)[0]
data_relevant = data_relevant.loc[mask]
data_relevant = data_relevant.reset_index(drop=True)

semester_length = len(ast.literal_eval(data_relevant['delta progress'][0]))

# %% transform trajectories

# normalise cumulative series
data_relevant['cumulative progress normalised'] = data_relevant.apply(
    normalize_cumulative_progress, axis=1)

# delta progress week wise
semester_length_weeks = round(semester_length/7)
data_relevant['delta progress weeks'] = data_relevant.apply(
    lambda row: process_delta_progress(row, semester_length_weeks), axis=1)

# cumulative progress week wise
data_relevant['cumulative progress weeks'] = data_relevant.apply(
    cumulative_progress_weeks, axis=1)

# choose columns to save
data_subset = data_relevant[['SUB_INDEX_194', 'Total credits',
                             'delta progress', 'cumulative progress',
                             'cumulative progress normalised',
                             'delta progress weeks',
                             'cumulative progress weeks']]

data_subset.to_csv('data_preprocessed.csv', index=False)

# %% cluster
timeseries_to_cluster = list(data_relevant.apply(
    get_timeseries_to_cluster, axis=1))

inertia = []
for cluster_size in range(1, 14):
    print(cluster_size+1)
    km = KMeans(n_clusters=cluster_size+1, n_init=10,
                random_state=0)
    labels = km.fit_predict(timeseries_to_cluster)
    inertia.append(km.inertia_)
plt.plot(inertia)

km = KMeans(n_clusters=3, n_init=10, random_state=0, verbose=True)
labels = km.fit_predict(timeseries_to_cluster)

help.plot_clustered_data(timeseries_to_cluster, labels)

data_clustered = pd.DataFrame(
    {'cumulative progress weeks': timeseries_to_cluster,
     'labels': labels})
# convert np floats to floats in rows
data_clustered["cumulative progress weeks"] = data_clustered[
    "cumulative progress weeks"].apply(
    lambda lst: [float(x) for x in lst])

data_clustered.to_csv('data_clustered.csv', index=False)

# %%
