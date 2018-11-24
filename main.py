import math
import time

from sklearn.cluster import AgglomerativeClustering, KMeans
from Kcluster import FuzzyKMeans
import Files
import pandas as pd
import scipy.stats.pearsonr

def print_clusters(X,labels):
    for i in range(len(X)):
        print("name:", X.iloc[[i]].index.values[0], "label:", labels[i])
        # plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)


def clean_null_figures(df: pd.DataFrame):
    df = df[(df.T != 0).any()]
    df = df.loc[:, (df != 0).any(axis=0)]
    return df


def log(x):
    if x == 0:
        return 0
    else:
        return math.log(x)


def count_values_bigger_than_k(s: pd.Series, k: int):
    count = 0
    for i in s:
        if i > k: count += 1
    return count

def clean_less_than_k(df: pd.DataFrame, value_threshold: int, rowcount_percentage:float):
    row_count = df.shape[0]
    row_threshold = rowcount_percentage*row_count
    #mask = pd.Series([count_values_bigger_than_k(s, value_threshold)>row_threshold for s in df.values])
    mask = pd.Series([count_values_bigger_than_k(s, value_threshold) > 10 for s in df.values],index=df.index)
    return df[mask]


if __name__ == '__main__':
    X: pd.DataFrame = pd.read_csv(open(Files.gold_set, 'rb'))
    X = X.set_index('Unnamed: 0')
    X = clean_null_figures(X)
    X = X.applymap(log)
    X = clean_less_than_k(X, 5,0.05)
    # X = X.fill(0).to_sparse(fill_value=0)
    AgglomerativeClustering(n_clusters=5,affinity=pearsonr).fit(X)
    #kmeans = FuzzyKMeans(k=4,)
    #kmeans.fit(X)
    #centroids = kmeans.cluster_centers_
    #labels = kmeans.labels_
    print_clusters(X,labels)
