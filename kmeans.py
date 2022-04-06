import os
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from statistics import mode
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image
from time import time
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier


def kmeans(X: np.ndarray, k: int, centroids=None, max_iter=30, tolerance=1e-2, verbose=False):
    # convert to float32 so that norm calclulation can canverage
    X = np.float64(X)
    if centroids == 'kmeans++':
        # centroids is k*p matrix where k is # centroids, p is dimensions
        # i.e: [[86.94],[80.65],[92.86]]
        print('!!!!activate kmeans++!!!!')
        m = select_centroids(X, k)

    else:
        # random choose k unique centriods
        m = X[np.random.choice(X.shape[0], k, replace=False), :]
    if verbose == 1:
        print('init centroids:')
        print(m)

    m_new = m.copy()

    # count iter
    t = 0

    for _ in tqdm(range(max_iter)):
        # if verbose == 1:
        #     print(f'inter: {t+1}')
        #     print('previous_centriod:')
        #     print(m)

        cluster = [[] for _ in range(k)]
        cluster_new = cluster.copy()

        j_star = []
        for cen in m:
            # j* is individual label
            j_star.append(np.linalg.norm(X-cen, axis=1))
            label = np.argmin(np.array(j_star).T, axis=1)

        # assign x to cluster
        for i in range(k):
            cluster[i].append(X[np.where(label == i)])

        # check if any cluster is empty
        for c in cluster:
            if len(c[0]) == 0:
                np.append(c[0],X[np.random.choice(X.shape[0]), :])

        # update m_new
        for index, _ in enumerate(m_new):
            # avg of data feature wise
            m_new[index] = np.mean(cluster[index][0], axis=0)
        # if verbose == 1:
        #     print('new_centriod:')
        #     print(m_new)

        j_star = []
        for cen in m_new:
            # j* is individual label
            j_star.append(np.linalg.norm(X-cen, axis=1))
            # print(j_star)
            label = np.argmin(np.array(j_star).T, axis=1)
        # assign x to cluster
        for i in range(k):
            cluster_new[i].append(X[np.where(label == i)])

        # compute avg_norm_cen
        avg_norm_cen = np.mean(np.linalg.norm(m-m_new, axis=1))
        # if verbose == 1:
        #     #print('new_cluster: ',cluster_new)
        #     print("=="*30)
        #     # print(m,'vs\n',m_new)
        #     print('average norm of centroids-previous_centroids:', avg_norm_cen)
        #     print("=="*30)

        t += 1

        if avg_norm_cen < tolerance or t >= max_iter:
            print('final norm: ', avg_norm_cen)
            if verbose == 1:
                print('final centroids: ')
                print(m_new)
                #print('label: ',np.array(label))

            return m_new, np.array(label)
        m = m_new.copy()


def select_centroids(X, k):
    """
    kmeans++ algorithm to select initial points:

    1. Pick first point randomly
    2. Pick next k-1 points by selecting points that maximize the minimum
       distance to all existing clusters. So for each point, compute distance
       to each cluster and find that minimum.  Among the min distances to a cluster
       for each point, find the max distance. The associated point is the new centroid.

    Return centroids as k x p array of points from X.
    """
    centroids = []
    # pick 1st center randomly
    m1 = X[np.random.choice(X.shape[0], replace=False), :]

    centroids.append(m1)
    # remove used point
    del_X = np.delete(X, np.where(X == m1), axis=0)
    # compute remaining k - 1 centroids
    for c_id in range(k-1):
        # compute distance of 'point' from each of the previously
        # selected centroid (del_X row wise) and store the minimum distance
        distances = np.argmax(np.linalg.norm(del_X-centroids[c_id], axis=1))
        next_centroid = del_X[distances, :]
        centroids.append(next_centroid)

        # remove used point
        del_X = np.delete(del_X, np.where(del_X == next_centroid), axis=0)
    return np.array(centroids).reshape(k, X.shape[1])


# support function: generate trees, X indice in the same tree means similar to each other
def leaf_samples(rf, X: np.ndarray):
    """
    Return a list of arrays where each array is the set of X sample indexes
    residing in a single leaf of some tree in rf forest. For example, if there
    are 4 leaves (in one or multiple trees), we might return:

        array([array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
               array([10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19, 20]),
               array([21, 22, 23, 24, 25, 26, 27, 28, 29]))
    """
    n_trees = len(rf.estimators_)
    leaf_samples = []
    leaf_ids = rf.apply(X)  # which leaf does each X_i go to for sole tree?
    for t in range(n_trees):
        # Group by id and return sample indexes
        uniq_ids = np.unique(leaf_ids[:, t])
        sample_idxs_in_leaves = [np.where(leaf_ids[:, t] == id)[
            0] for id in uniq_ids]
        leaf_samples.extend(sample_idxs_in_leaves)

    # need n_trees in similarity_matrix
    return leaf_samples, n_trees


def similarity_matrix(X, rf):
    leaves, n_trees = leaf_samples(rf, X)
    #nt = pd.DataFrame(trees).T
    ept_matrix = np.zeros((X.shape[0], X.shape[0]))
    for leaf in leaves:
        for row_index in leaf:
            # every time same pair of records appear in different trees add 1
            ept_matrix[row_index, leaf] += 1
    return ept_matrix/n_trees

# support function generate a confusion matrix and accuracy


def likely_confusion_matrix(y, labels):
    if labels.sum()/y.sum() < .6:
        # flip the label of predicted labels
        labels = 1 - labels

    ConfusionMatrixDisplay.from_predictions(y, labels)
    print(f'clustering accur: {(labels==y).mean()}')


def comp_adv():
    # circle
    X, y = make_circles(n_samples=500, noise=0.1, factor=.2)

    cluster = SpectralClustering(n_clusters=2, affinity="nearest_neighbors")
    labels = cluster.fit_predict(X)  # pass X not similarity matrix

    colors = np.array(['#4574B4', '#A40227'])
    plt.scatter(X[:, 0], X[:, 1], c=colors[labels])
    plt.title('sklearn SpectralClustering')
    plt.show()

    rf = RandomForestClassifier()
    rf.fit(X, y)
    S = similarity_matrix(X, rf)  # breiman's trick
    cluster = SpectralClustering(n_clusters=2, affinity='precomputed')
    label = cluster.fit_predict(S)  # pass similarity matrix not X
    colors = np.array(['#4574B4', '#A40227'])
    plt.scatter(X[:, 0], X[:, 1], c=colors[label])
    plt.title('RF+Kmeans')
    plt.show()
