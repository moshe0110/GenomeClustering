import time

from sklearn.cluster import AgglomerativeClustering, KMeans

import Files
import pandas as pd
from sklearn import cluster,neighbors
import networkx as nx
from collections import defaultdict
import matplotlib

#matplotlib.use('TkAgg')
from matplotlib import cm
import seaborn as sns
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

import matplotlib.pyplot as plt

def project1 ():
    df = pd.read_csv(open(Files.grover_5mb, 'rb'))
    df = df.fillna(0).to_sparse(fill_value=0)
    # res = sklearn.cluster.k_means(df, 10)


def print_clusters(X):
    for i in range(len(X)):
        print("name:", X[i][0], "label:", labels[i])
        #plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

if __name__ == '__main__':
    # X = pd.DataFrame([[1, 2, 30],
    #                   [5, 8, 3],
    #                   [1.5, 1.8, 3],
    #                   [8, 8, 30],
    #                   [1, 0.6, 30],
    #                   [9, 11, 3]])

    X = pd.read_csv(open(Files.grover_5mb, 'rb'))
    X = X.set_index('Unnamed: 0')

    #X = X.fill(0).to_sparse(fill_value=0)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    X = X.values

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_


    print_clusters(X)


    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)

    plt.show()
