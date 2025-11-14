# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import ftfy
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
import constants
import matplotlib as mpl
mpl.rcParams['font.size'] = 18

# %%


def get_correlation(a, b):
    mask = np.isnan(a)
    a = a[~mask]
    b = b[~mask]
    mask2 = np.isnan(b)
    a = a[~mask2]
    b = b[~mask2]
    print(pearsonr(a, b))
    plt.figure()
    plt.scatter(a, b)


def get_completion_week(row):
    hours = np.array(ast.literal_eval(
        row['cumulative progress weeks']))
    if np.max(hours) >= 7:
        return np.where(hours >= 7)[0][0]
    else:
        return np.nan


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


def safe_fix(text):
    if isinstance(text, str):
        return ftfy.fix_text(text)
    return text


# %% import data
data_relevant = pd.read_csv('data_preprocessed.csv', index_col=False)

data_full = pd.read_csv('zhang_ma_data.csv',
                        index_col=False)

result_fit_mle = np.load(
    "fits/fit_individual_mle.npy", allow_pickle=True)

# result_fit_em = np.load("fits/fit_pop_em.npy", allow_pickle=True).item()

# %%
nllkhd = sum([result_fit_mle[i]['neg_log_lik']
              for i in range(len(result_fit_mle))])
BIC = BIC_mle = 2*nllkhd + \
    (len(result_fit_mle) * 3 * np.log(constants.HORIZON))
# %%
data_full_filter = data_full[data_full['SUB_INDEX_194'].isin(
    data_relevant['SUB_INDEX_194'])]
# result_fit_params = np.vstack(np.hstack(result_fit_mle[:, 1, :]))
result_fit_params = np.array([result_fit_mle[i]['par_b']
                              for i in range(len(result_fit_mle))])

# np.save('fits/fit_params_mle_extended_ranges.npy',
#         result_fit_params, allow_pickle=True)

for i in range(3):
    plt.figure(figsize=(4, 4))
    plt.hist(result_fit_params[:, i])

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

# %%
fit_params_recoverable = np.load('fits/fit_params_mle_recoverable.npy',
                                 allow_pickle=True)
data_recoverable = pd.read_csv('data_recoverable.csv', index_col=False)
data_processed = pd.read_csv(
    'data_preprocessed.csv', index_col=False)
data_processed_recoverable = pd.read_csv(
    'data_preprocessed_recoverable.csv', index_col=False)
# %%

fit_params = result_fit_params  # fit_params_recoverable
data = data_full_filter  # data_recoverable

discount_factors_log_empirical = np.array(data['DiscountRate_lnk'])
discount_factors_empirical = np.exp(discount_factors_log_empirical)
discount_factors_fitted = fit_params[:, 0]
efficacy_fitted = fit_params[:, 1]
efforts_fitted = fit_params[:, 2]
proc_mean = np.array(data['AcadeProcFreq_mean'])
impulsivity_score = np.array(data['ImpulsivityScore'])
time_management = np.array(data['ReasonProc_TimeManagement'])
task_aversiveness = np.array(data['ReasonProc_TaskAversiveness'])
laziness = np.array(data['ReasonProc_Laziness'])
self_control = np.array(data['SelfControlScore'])

discount_factors_empirical = np.exp(discount_factors_log_empirical)
# correlations from paper
get_correlation(discount_factors_log_empirical, discount_factors_fitted)
get_correlation(discount_factors_empirical, discount_factors_fitted)
get_correlation(proc_mean, discount_factors_fitted)
get_correlation(proc_mean, efficacy_fitted)
get_correlation(proc_mean, efforts_fitted)
get_correlation(impulsivity_score, discount_factors_fitted)
get_correlation(self_control, discount_factors_fitted)
get_correlation(task_aversiveness, efforts_fitted)
get_correlation(laziness, efforts_fitted)
get_correlation(time_management, efficacy_fitted)

# %% task based measures
data_p = data_processed  # data_processed_recoverable

completion_week = np.array(data_p.apply(get_completion_week, axis=1))
mucw = np.array(data_p.apply(get_mucw, axis=1))

delay = mucw  # completion_week

get_correlation(completion_week, mucw)

df = pd.DataFrame({'delay': delay,
                   'disc': discount_factors_fitted,
                   'efficacy': efficacy_fitted,
                   'effort': efforts_fitted})
df = df.dropna()
cols_to_z_score = ['disc', 'efficacy', 'effort']
df_z_scored = df.copy()
df_z_scored[cols_to_z_score] = (
    (df_z_scored[cols_to_z_score]
     - df_z_scored[cols_to_z_score].mean())
    / df_z_scored[cols_to_z_score].std())
model = smf.ols(
    formula='delay ~ disc + efficacy + effort',
    data=df_z_scored).fit()

print(model.summary())

# %%
# correlations from paper
delay = mucw  # completion_week
get_correlation(proc_mean, delay)
get_correlation(discount_factors_log_empirical, delay)
get_correlation(discount_factors_empirical, delay)
get_correlation(discount_factors_log_empirical, proc_mean)
get_correlation(discount_factors_empirical, proc_mean)

# %%
df = pd.DataFrame({'PASS': proc_mean,
                   'disc': discount_factors_fitted,
                   'efficacy': efficacy_fitted,
                   'effort': efforts_fitted})
df = df.dropna()
cols_to_z_score = ['disc', 'efficacy', 'effort']
df_z_scored = df.copy()
df_z_scored[cols_to_z_score] = (
    (df_z_scored[cols_to_z_score]
     - df_z_scored[cols_to_z_score].mean())
    / df_z_scored[cols_to_z_score].std())
model = smf.ols(
    formula='PASS ~ disc + efficacy + effort',
    data=df_z_scored).fit()

print(model.summary())

# %% text responses

df = data_processed.set_index('SUB_INDEX_194')

# planners
idxs = [4, 18, 21, 109]
# workload
idxs = [64, 86, 147, 181]
# lack of motivation
idxs = [0, 25, 58, 110]
# forgetting
idxs = [7, 52, 157]
for idx in idxs:
    plt.plot(ast.literal_eval(df.loc[idx, 'cumulative progress weeks']),
             color='gray')
plt.xlabel('weeks')
plt.ylabel('cumulative hours finished')

# %% sentence embeddings

data_full_filter['TextReport_cause_procrastination'] = data_full_filter[
    'TextReport_cause_procrastination'].apply(safe_fix)
data_full_filter.loc[data_full_filter[
    'TextReport_cause_procrastination'] == 'N/a'] = np.nan

model = SentenceTransformer('all-mpnet-base-v2')  # all-MiniLM-L6-v2
responses = data_full_filter[
    'TextReport_cause_procrastination'].dropna().tolist()
embeddings = model.encode(responses, show_progress_bar=True)

# k-means
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
labels_kmeans = kmeans.fit_predict(embeddings)

# hierarchical clustering
distance_matrix = squareform(pdist(embeddings, metric='cosine'))
Z = linkage(distance_matrix, method='complete')
plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    leaf_rotation=90,
    leaf_font_size=10,
    color_threshold=3.2
)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Responses")
plt.ylabel("Distance")
plt.show()
# t=0.7 * max(Z[:, 2])
labels_hier = fcluster(Z, t=3.2,  criterion='distance')

# group responses by labels
clusters = defaultdict(list)
for response, label in zip(responses, labels_hier):
    clusters[label].append(response)

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    max_iter=1000,
    metric='cosine',
    random_state=0
)
embeddings_2d = tsne.fit_transform(embeddings)

# PCA
pca = PCA(n_components=2)
points_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_hier)
plt.xlabel("dim 1")
plt.ylabel("dim 2")
plt.show()

# %%
