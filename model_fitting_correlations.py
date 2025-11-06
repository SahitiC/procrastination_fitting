# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import ftfy
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict

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
data_full_filter = data_full[data_full['SUB_INDEX_194'].isin(
    data_relevant['SUB_INDEX_194'])]
# result_fit_params = np.vstack(np.hstack(result_fit_mle[:, 1, :]))
result_fit_params = np.array([result_fit_mle[i]['par_b']
                              for i in range(len(result_fit_mle))])

# np.save('fits/fit_params_mle_basic_lite.npy',
#         result_fit_params, allow_pickle=True)

for i in range(3):
    plt.figure(figsize=(4, 4))
    plt.hist(result_fit_params[:, i])

# %%

discount_factors_log_empirical = np.array(data_full_filter['DiscountRate_lnk'])
discount_factors_fitted = result_fit_params[:, 0]
efficacy_fitted = result_fit_params[:, 1]
efforts_fitted = result_fit_params[:, 2]
proc_mean = np.array(data_full_filter['AcadeProcFreq_mean'])
impulsivity_score = np.array(data_full_filter['ImpulsivityScore'])
time_management = np.array(data_full_filter['ReasonProc_TimeManagement'])
task_aversiveness = np.array(data_full_filter['ReasonProc_TaskAversiveness'])

discount_factors_empirical = np.exp(discount_factors_log_empirical)
get_correlation(discount_factors_log_empirical, discount_factors_fitted)
get_correlation(proc_mean, discount_factors_fitted)
get_correlation(impulsivity_score, discount_factors_fitted)
get_correlation(task_aversiveness, np.abs(efforts_fitted))
get_correlation(time_management, efficacy_fitted)

# %% task based measures
data_processed = pd.read_csv(
    'data_preprocessed.csv', index_col=False)

completion_week = np.array(data_processed.apply(get_completion_week, axis=1))

delay = completion_week

df = pd.DataFrame({'delay': completion_week,
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
