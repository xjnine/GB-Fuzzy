# -*- coding: utf-8 -*-

import numpy as np

def FCM_no_random(X, c_clusters=2, m=2, eps=10, max_count=100):
    membership_mat = np.random.random((len(X), c_clusters))
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])
    center = X.mean(0)
    p1 = np.array(0)
    p2 = np.array(0)
    n, d = np.shape(X)
    dist_1 = np.sqrt(np.sum(np.asarray(center - X) ** 2, axis=1))
    index_1 = np.where(dist_1 == np.max(dist_1))
    p1 = np.reshape(X[index_1[0][0], :], [d, ])
    dist_2 = np.sqrt(np.sum(np.asarray(p1 - X) ** 2, axis=1))
    index_2 = np.where(dist_2 == np.max(dist_2))
    p2 = np.reshape(X[index_2[0][0], :], [d, ])
    c_p1 = (center + p1) / 2
    c_p2 = (center + p2) / 2
    r = np.sqrt(np.sum(np.asarray(c_p1 - c_p2) ** 2, axis=0)) / 2
    flag = False
    Centroids = np.array([c_p1, c_p2])
    working_membership_mat = membership_mat ** m
    count = 0
    while True:
        n_c_distance_mat = np.zeros((len(X), c_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x - c, 2)
        new_membership_mat = np.zeros((len(X), c_clusters))
        for i, x in enumerate(X):
            if x.tolist() in Centroids.tolist():
                if (x == Centroids[0]).all():
                    new_membership_mat[i][0] = 1
                    new_membership_mat[i][1] = 0
                else:
                    new_membership_mat[i][1] = 1
                    new_membership_mat[i][0] = 0
                break
            for j, c in enumerate(Centroids):
                new_membership_mat[i][j] = 1. / np.sum(
                    (n_c_distance_mat[i][j] / (n_c_distance_mat[i])) ** (2 / (m - 1)))
        flag = True
        if count > max_count or np.sum(abs(new_membership_mat - membership_mat)) < eps:
            break
        else:
            count += 1
        membership_mat = new_membership_mat
        working_membership_mat = membership_mat ** m
        Centroids = np.divide(np.dot(working_membership_mat.T, X),
                              np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])
    return np.argmax(new_membership_mat, axis=1)
