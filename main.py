import math
import time

from sklearn.cluster import AgglomerativeClustering, KMeans
from Kcluster import FuzzyKMeans
import Files
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt



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

    mask = pd.Series([count_values_bigger_than_k(s, value_threshold) > row_threshold for s in df.values],index=df.index)

    return df[mask]

def plot_heatmap_sorted(clustering: list, X):
    sorted_X = pd.DataFrame()
    for i in range(0, NUM_OF_CLUSTERS):
        indicators = list(map(lambda j: j == i, clustering))
        cluster_i: list = X[indicators]
        sorted_X = pd.concat([sorted_X, cluster_i])

    fig, ax = plt.subplots(figsize=(10, 15))
    sns.heatmap(sorted_X, ax=ax)
    plt.show()
if __name__ == '__main__':
    NUM_OF_CLUSTERS=[5,7,11]
    CLEANING_PERCENTAGES =  [0.00075,0.001,0.0015]

    X: pd.DataFrame = pd.read_csv(open(Files.gold_set, 'rb'))
    X = X.set_index('Unnamed: 0')
    X = clean_null_figures(X)
    X = X.applymap(log)
    for clean_percentage in CLEANING_PERCENTAGES:
        for num_of_clusters in NUM_OF_CLUSTERS:
            print('starting run')
            print(f'num of clusters: {num_of_clusters}')
            print(f'clean_percentage: {clean_percentage}')
            X = clean_less_than_k(X, 3, clean_percentage)
            X_np:np = np.asarray(X)
            X_np_corr:np = np.corrcoef(X_np)
            X_computed_affinity:np = np.asanyarray([np.array([1 - xi for xi in x]) for x in X_np_corr])
            print('started ')
            clustering_complete = AgglomerativeClustering(n_clusters=num_of_clusters, affinity='precomputed', linkage='complete').fit_predict(X_computed_affinity)

            print('finished clustering')






'''
useful code:


In [4]: frames = [df1, df2, df3]

In [5]: result = pd.concat(frames)

    X = X.fill(0).to_sparse(fill_value=0)
    corr = X.T.corr()
    corr = corr.applymap(lambda x:1-x)
    affinty1 = lambda A,B:np.corrcoef(A, B)[0,1]
    matrix_affinty = lambda K:K.T.corr()
    kmeans = FuzzyKMeans(k=4,)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
'''

