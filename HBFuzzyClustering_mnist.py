# -*- coding: utf-8 -*-

import time

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import FuzzyClustering_no_random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import k_means



class GB:
    def __init__(self, data,
                 label):  # Data is labeled data, the penultimate column is label, and the last column is index
        self.data = data
        self.center = self.data.mean(
            0)  # According to the calculation of row direction, the mean value of all the numbers in each column (that is, the center of the pellet) is obtained
        # self.init_center = self.random_center()  # Get a random point in each tag
        self.radius = self.get_radius()
        self.flag = 0
        self.label = label
        self.num = len(data)
        self.out = 0
        self.size = 1
        self.overlap = 0  # 默认0的时候使用软重叠，1使用硬重叠
        self.hardlapcount = 0
        self.softlapcount = 0

    def get_radius(self):
        return max(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)


class UF:
    def __init__(self, len):
        self.parent = [0] * len
        self.size = [0] * len
        self.count = len

        for i in range(0, len):
            self.parent[i] = i
            self.size[i] = 1

    def find(self, x):
        while (self.parent[x] != x):
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if (rootP == rootQ):
            return
        if self.size[rootP] > self.size[rootQ]:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        else:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        self.count = self.count - 1

    def connected(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ

    def count(self):
        return self.count


def division(hb_list, hb_list_not):
    gb_list_new = []
    for hb in hb_list:
        if len(hb) > 0:
            ball_1, ball_2 = spilt_ball_fuzzy(hb)
            dm_parent = get_dm(hb)
            dm_child_1 = get_dm(ball_1)
            dm_child_2 = get_dm(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = w1 * dm_child_1 + w2 * dm_child_2
            t2 = w_child < dm_parent
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                hb_list_not.append(hb)
        else:
            hb_list_not.append(hb)
    return gb_list_new, hb_list_not


def spilt_ball_fuzzy(data):
    ball1 = []
    ball2 = []
    cluster = FuzzyClustering_no_random.FCM_no_random(data, 2)
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]


def get_dm(hb):
    num = len(hb)
    if num == 0:
        return 1
    center = hb.mean(0)
    diff_mat = np.tile(center, (num, 1)) - hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sum_radius = 0
    for i in distances:
        sum_radius = sum_radius + i
    if num > 2:
        mean_radius = sum_radius / num
        return mean_radius
    else:
        return 1


def get_radius(hb):
    num = len(hb)
    center = hb.mean(0)
    diff_mat = np.tile(center, (num, 1)) - hb
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    radius = max(distances)
    return radius


def plot_dot(data):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], s=7, c="#314300", linewidths=5, alpha=0.6, marker='o', label='data point')
    plt.legend(loc=1)


def hb_plot_3d(hbs, ax, labelsize):
    color = {
        0: '#fb8e94',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#707afa',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#e96d29',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',
        25: '#ff8444', }
    label_c = {
        0: 'cluster-1',
        1: 'cluster-2',
        2: 'cluster-3',
        3: 'cluster-4',
        4: 'cluster-5',
        5: 'cluster-6',
        6: 'cluster-7',
        7: 'cluster-8',
        8: 'cluster-9',
        9: 'cluster-10',
        10: 'cluster-11',
        11: 'cluster-12',
        12: 'cluster-13',
        13: 'cluster-14',
        14: 'cluster-15',
        15: 'cluster-16',
        16: 'cluster-17',
        17: 'cluster-18',
        18: 'cluster-19',
        19: 'cluster-20',
        20: 'cluster-21',
        21: 'cluster-22',
        22: 'cluster-23',
        23: 'cluster-24',
        24: 'cluster-25'}

    label_num = {}
    for i in range(0, len(hbs)):
        label_num.setdefault(hbs[i].label, 0)
        label_num[hbs[i].label] = label_num.get(hbs[i].label) + len(hbs[i].data)

    label = set()
    for key in label_num.keys():
        if label_num[key] > 10:
            label.add(key)
    list = []
    for i in range(0, len(label)):
        list.append(label.pop())

    for key in hbs.keys():
        for i in range(0, len(list)):
            if (hbs[key].label == list[i]):
                if (i < 36):
                    ax.scatter(hbs[key].data[:, 0], hbs[key].data[:, 1], hbs[key].data[:, 2], s=10, c=color[i],
                               linewidths=3.3, alpha=0.8)
                break

    for i in range(0, len(list)):
        for key in hbs.keys():
            if (hbs[key].label == list[i]):
                ax.scatter(hbs[key].data[:, 0], hbs[key].data[:, 1], hbs[key].data[:, 2], s=6, c=color[i],
                           linewidths=3.3, alpha=0.8, marker='o', label=label_c[i])
                break
    plt.tick_params(labelsize=labelsize)
    # ax.legend(loc=2, numpoints=1, fontsize=32)


def hb_plot_3d_2(data, clusters, nclusters, ax, labelsize):
    color = {
        0: '#fb8e94',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#707afa',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#e96d29',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',
        25: '#ff8444', }
    label_c = {
        0: 'cluster-1',
        1: 'cluster-2',
        2: 'cluster-3',
        3: 'cluster-4',
        4: 'cluster-5',
        5: 'cluster-6',
        6: 'cluster-7',
        7: 'cluster-8',
        8: 'cluster-9',
        9: 'cluster-10',
        10: 'cluster-11',
        11: 'cluster-12',
        12: 'cluster-13',
        13: 'cluster-14',
        14: 'cluster-15',
        15: 'cluster-16',
        16: 'cluster-17',
        17: 'cluster-18',
        18: 'cluster-19',
        19: 'cluster-20',
        20: 'cluster-21',
        21: 'cluster-22',
        22: 'cluster-23',
        23: 'cluster-24',
        24: 'cluster-25'}

    for i in range(0, nclusters):
        ax.scatter(data[clusters == i, :][:, 0], data[clusters == i, :][:, 1], data[clusters == i, :][:, 2], s=10,
                   c=color[i], linewidths=3.3, alpha=0.8, marker='o', label=label_c[i])

    plt.tick_params(labelsize=labelsize)
    # ax.legend(loc=2, numpoints=1, fontsize=32)


def draw_3d_ball(hb_list, ax, data, labels_true, labelsize):
    color = {0: '#CCCCFF', 1: '#AFDCEC', 2: '#7BCCB5', 3: '#FFCBA4', 4: '#FAAFBE'}
    count = 0
    # for i in range(0, len(data)):
    #     ax.text(data[i, 0], data[i, 1], data[i, 2], str(labels_true[i]), c='#000000',
    #             fontdict={'weight': 'light', 'size': 15})
    for data in hb_list:
        count = count + 1
        if len(data) > 1:
            center_2d = data.mean(0)
            radius = np.max((((data - center_2d) ** 2).sum(axis=1) ** 0.5))
            center_3d = np.append(center_2d, 0)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center_3d[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center_3d[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_3d[2]
            ax.plot_wireframe(x, y, z, rstride=4, cstride=4, alpha=0.1, color=color[count % 5])

    plt.tick_params(labelsize=labelsize)
    # ax.legend(loc=1, numpoints=1, fontsize=12)


def connect_ball_kmeans(hb_list, noise, nclusters, data):
    hb_center_list = []
    for i in range(0, len(hb_list)):
        hb_center_list.append(hb_list[i].mean(0))
    center_data = np.array(hb_center_list)
    clusters_center = k_means(X=center_data, init='k-means++', n_clusters=nclusters)[1]
    index = np.full((1, data.shape[0]), 0).T
    for j in range(0, len(hb_list)):
        index_temp = np.unique(np.where(data == hb_list[j][:, None])[1])
        index[index_temp] = clusters_center[j]
    clusters = np.reshape(index, [len(data), ])

    return clusters


def connect_ball_overlap(hb_list, noise, c_count):
    hb_cluster = {}
    for i in range(0, len(hb_list)):
        hb = GB(hb_list[i], i)
        hb_cluster[i] = hb

    radius_sum = 0
    num_sum = 0
    gblen = 0
    radius_sum = 0
    num_sum = 0
    for i in range(0, len(hb_cluster)):
        if hb_cluster[i].out == 0:
            gblen = gblen + 1
            radius_sum = radius_sum + hb_cluster[i].radius
            num_sum = num_sum + hb_cluster[i].num

    for i in range(0, len(hb_cluster) - 1):
        if hb_cluster[i].out != 1:
            center_i = hb_cluster[i].center
            radius_i = hb_cluster[i].radius
            for j in range(i + 1, len(hb_cluster)):
                if hb_cluster[j].out != 1:
                    center_j = hb_cluster[j].center
                    radius_j = hb_cluster[j].radius
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if (dis <= radius_i + radius_j) & ((hb_cluster[i].hardlapcount == 0) & (
                            hb_cluster[j].hardlapcount == 0)):  # 由于两个的点是噪声球，所以纳入重叠统计
                        hb_cluster[i].overlap = 1
                        hb_cluster[j].overlap = 1
                        hb_cluster[i].hardlapcount = hb_cluster[i].hardlapcount + 1
                        hb_cluster[j].hardlapcount = hb_cluster[j].hardlapcount + 1

    hb_uf = UF(len(hb_list))
    for i in range(0, len(hb_cluster) - 1):
        if hb_cluster[i].out != 1:
            center_i = hb_cluster[i].center
            radius_i = hb_cluster[i].radius
            for j in range(i + 1, len(hb_cluster)):
                if hb_cluster[j].out != 1:
                    center_j = hb_cluster[j].center
                    radius_j = hb_cluster[j].radius
                    max_radius = max(radius_i, radius_j)
                    min_radius = min(radius_i, radius_j)
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if (c_count == 1):
                        dynamic_overlap = dis <= radius_i + radius_j + 1 * (min_radius) / (
                                min(hb_cluster[i].hardlapcount, hb_cluster[j].hardlapcount) + 1)
                    if (c_count == 2):
                        dynamic_overlap = dis <= radius_i + radius_j + 1 * (max_radius) / (
                                min(hb_cluster[i].hardlapcount, hb_cluster[j].hardlapcount) + 1)
                    num_limit = ((hb_cluster[i].num > 2) & (hb_cluster[j].num > 2))
                    if dynamic_overlap & num_limit:
                        hb_cluster[i].flag = 1
                        hb_cluster[j].flag = 1
                        hb_uf.union(i, j)
                    if dis <= radius_i + radius_j + ((max_radius)):
                        hb_cluster[i].softlapcount = 1
                        hb_cluster[j].softlapcount = 1

    for i in range(0, len(hb_cluster)):
        k = i
        if hb_uf.parent[i] != i:
            while (hb_uf.parent[k] != k):
                k = hb_uf.parent[k]
        hb_uf.parent[i] = k

    for i in range(0, len(hb_cluster)):
        hb_cluster[i].label = hb_uf.parent[i]
        hb_cluster[i].size = hb_uf.size[i]

    label_num = set()
    for i in range(0, len(hb_cluster)):
        label_num.add(hb_cluster[i].label)

    list = []
    for i in range(0, len(label_num)):
        list.append(label_num.pop())

    for i in range(0, len(hb_cluster)):
        if ((hb_cluster[i].hardlapcount == 0) & (hb_cluster[i].softlapcount == 0)):
            hb_cluster[i].flag = 0

    for i in range(0, len(list)):
        count = 0
        list1 = []
        for key in range(0, len(hb_cluster)):
            if hb_cluster[key].label == list[i]:
                count += 1
                list1.append(key)
        while count < 4:
            for j in range(0, len(list1)):
                hb_cluster[list1[j]].flag = 0
            break

    for i in range(0, len(hb_cluster)):
        distance = np.sqrt(2)
        if hb_cluster[i].flag == 0:
            for j in range(0, len(hb_cluster)):
                if hb_cluster[j].flag == 1:
                    center = hb_cluster[i].center
                    center2 = hb_cluster[j].center
                    dis = ((center - center2) ** 2).sum(axis=0) ** 0.5 - (hb_cluster[i].radius + hb_cluster[j].radius)
                    if dis < distance:
                        distance = dis
                        hb_cluster[i].label = hb_cluster[j].label
                        hb_cluster[i].flag = 2
            for k in range(0, len(noise)):
                center = hb_cluster[i].center
                dis = ((center - noise[k]) ** 2).sum(axis=0) ** 0.5
                if dis < distance:
                    distance = dis
                    hb_cluster[i].label = -1
                    hb_cluster[i].flag = 2

    label_num = set()
    for i in range(0, len(hb_cluster)):
        label_num.add(hb_cluster[i].label)
    return hb_cluster


def normalized_ball(hb_list, hb_list_not, radius_detect):
    hb_list_temp = []
    for hb in hb_list:
        if len(hb) < 2:
            hb_list_not.append(hb)
        else:
            if get_radius(hb) <= 2 * radius_detect:
                hb_list_not.append(hb)
            else:
                # ball_1, ball_2 = spilt_ball(hb)
                ball_1, ball_2 = spilt_ball_fuzzy(hb)
                hb_list_temp.extend([ball_1, ball_2])

    return hb_list_temp, hb_list_not


def hbc_fuzzy():
    keys_1 = ['mnist_data']
    keys_2 = ['mnist_label']
    for d in range(len(keys_1)):
        df_1 = pd.read_csv(r"./mnist/" + keys_1[d] + ".csv", header=None)
        df_2 = pd.read_csv(r"./mnist/" + keys_2[d] + ".csv", header=None)
        data = df_1.values
        labels_true = df_2.values
        labels_true = labels_true.reshape(len(labels_true), )
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        index = np.full((1, data.shape[0]), 0)
        data_index = np.insert(data, data.shape[1], values=index, axis=1)
        hb_list_temp = [data]
        hb_list_not_temp = []
        row = np.shape(hb_list_temp)[0]
        col = np.shape(hb_list_temp)[1]
        n = row * col
        while 1:
            ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
            hb_list_temp, hb_list_not_temp = division(hb_list_temp, hb_list_not_temp)
            ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
            if ball_number_new == ball_number_old:
                hb_list_temp = hb_list_not_temp
                break
        radius = []
        for hb in hb_list_temp:
            if len(hb) >= 2:
                radius.append(get_radius(hb))
        radius_median = np.median(radius)
        radius_mean = np.mean(radius)
        radius_detect = max(radius_median, radius_mean)
        hb_list_not_temp = []
        labelsize = 55
        while 1:
            ball_number_old = len(hb_list_temp) + len(hb_list_not_temp)
            hb_list_temp, hb_list_not_temp = normalized_ball(hb_list_temp, hb_list_not_temp, radius_detect)
            ball_number_new = len(hb_list_temp) + len(hb_list_not_temp)
            if ball_number_new == ball_number_old:
                hb_list_temp = hb_list_not_temp
                break
        fig = plt.figure(figsize=(14.5, 14.5))
        ax = Axes3D(fig)
        draw_3d_ball(hb_list_temp, ax, data, labels_true, labelsize)
        plt.show()
        # FGB-overlap
        noise = []
        hb_cluster = connect_ball_overlap(hb_list_temp, noise, 1)
        end_time = time.time()
        for i in range(0, len(hb_cluster)):
            for j in range(0, len(data)):
                if ((data[j] in hb_cluster[i].data)):
                    data_index[j, data.shape[1]] = hb_cluster[i].label
        labels_pred = (data_index[:, data.shape[1]]).T
        data = data.tolist()
        final_label = [0] * len(data)
        final_label2 = [0] * len(data)
        for i in range(0, len(hb_cluster)):
            for j in hb_cluster[i].data.tolist():
                final_label[data.index(j)] = hb_cluster[i].label
        label_num = set()
        for i in range(0, len(hb_cluster)):
            label_num.add(hb_cluster[i].label)
        label_sort = list(label_num)
        label_sort.sort()
        label_dict = {}
        for i in range(len(label_sort)):
            label_dict[str(label_sort[i])] = i
        for i in range(len(final_label)):
            if label_dict.keys().__contains__(str(final_label[i])):
                final_label2[i] = label_dict[str(final_label[i])]
        data = np.array(data)
        final_label2 = np.array(final_label2)
        labels_pred = final_label2
        fig = plt.figure(figsize=(14.5, 14.5))
        ax = Axes3D(fig)
        hb_plot_3d(hb_cluster, ax, labelsize)
        plt.show()

        # FGB-KMeans
        # noise = []
        # nclusters = 10
        # # hb_cluster = connect_ball_fuzzy(hb_list_temp,noise,nclusters,data)
        # hb_cluster = connect_ball_kmeans(hb_list_temp, noise, nclusters, data)
        # labels_pred = hb_cluster
        # df = pd.DataFrame(labels_true)
        # # df.to_csv(r'.\clusters.csv', index=None, header=None)
        # end_time = time.time()
        # fig = plt.figure(figsize=(14.5, 14.5))
        # ax = Axes3D(fig)
        # # hb_cluster = pd.read_csv('mnist/res/K-means/clusters.csv', header=None).values.reshape(len(labels_true), )
        # hb_plot_3d_2(data, hb_cluster, nclusters, ax,labelsize)
        # plt.show()


if __name__ == '__main__':
    hbc_fuzzy()
