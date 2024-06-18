import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
def additive_utilitarian(cluster_mat):
    return np.sum(cluster_mat, axis=0)
def average(cluster_mat):
    return np.mean(cluster_mat, axis=0)
def simple_count(cluster_mat):
    return np.count_nonzero(cluster_mat, axis=0)
def approval_voting(cluster_mat):
    return np.sum(cluster_mat >= 4, axis=0)
def borda_count(cluster_mat):
    ranks = np.argsort(np.argsort(-cluster_mat, axis=1), axis=1)
    count = np.sum(ranks, axis=0)
    return count
def copeland_rule(cluster_mat):
    items = cluster_mat.shape[1]
    wins = np.zeros(items)
    for i in range(items):
        for j in range(items):
            if i != j:
                wins[i] += np.sum(cluster_mat[:, i] > cluster_mat[:, j])
                wins[i] -= np.sum(cluster_mat[:, i] < cluster_mat[:, j])
    return wins

file_path = 'C:/Temp/ratings.dat'
num_users = 6040
num_movies = 3952
X_data = np.zeros((num_users, num_movies))

df = pd.read_csv(file_path, delimiter='::', engine='python', header=None, names=['UserID', 'MovieID', 'Rating'])

for row in df.itertuples():
    X_data[row.UserID - 1, row.MovieID - 1] = row.Rating

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
X_train = kmeans.fit_predict(X_data)

df_clusters = pd.DataFrame({'UserID': np.arange(1, len(X_train) + 1), 'Cluster': X_train})

recommendation = {}

for id in range(3):
    indices = df_clusters[df_clusters['Cluster'] == id].index - 1  # 인덱스를 0부터 시작하게 조정
    cluster_mat = X_data[indices]

    au = additive_utilitarian(cluster_mat)
    avg = average(cluster_mat)
    sc = simple_count(cluster_mat)
    av = approval_voting(cluster_mat)
    bc = borda_count(cluster_mat)
    cr = copeland_rule(cluster_mat)

    recommendation[id] = {
        'AU': np.argsort(-au)[:10],
        'Avg': np.argsort(-avg)[:10],
        'SC': np.argsort(-sc)[:10],
        'AV': np.argsort(-av)[:10],
        'BC': np.argsort(-bc)[:10],
        'CR': np.argsort(-cr)[:10],
    }

for id, reco in recommendation.items():
    print(f"Group {id + 1}:")
    for method, items in reco.items():
        print(f"{method} Top 10: {items}")
    print("\n")
