# %%
"""
pre-process data from Zhang and Ma (2024)
"""
import numpy as np
import pandas as pd
import ast

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
    """
    get cumulative progress in weeks from delta progress in weeks
    """
    return list(np.cumsum(row['delta progress weeks']))


# %% drop unwanted rows

data = pd.read_csv('data/zhang_ma_data.csv')

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

data_subset.to_csv('data/data_preprocessed.csv', index=False)
