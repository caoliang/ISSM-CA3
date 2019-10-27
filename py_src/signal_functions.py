'''
def pltDistances(dists, title, xlab="X", ylab="Y", clrmap="viridis"):
    #imgplt = plt.figure(figsize=(4, 4))
    plt.suptitle(title, fontsize=20)
    plt.imshow(dists, interpolation='nearest', cmap=clrmap)
    plt.gca().invert_yaxis()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid()
    plt.colorbar()

    return imgplt

def plt_cost_and_path(acuCost, path, title, xlab="X", ylab="Y", clrmap="viridis"):
    px = [pt[0] for pt in path]
    py = [pt[1] for pt in path]
    imgplt = pltDistances(acuCost,
                          title,
                          xlab=xlab,
                          ylab=ylab,
                          clrmap=clrmap)
    plt.plot(px, py)
    return imgplt

'''

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.random import seed
from tensorflow import set_random_seed

def draw_time_series(ts_data_x, ts_data_y, index_start, length):
    plt.figure()
    oven_cols_index = ts_data_x.shape[1] - 1
    n_parts = 5
    for i in range(index_start, index_start + length):
        fig, axes = plt.subplots(1, n_parts, figsize=(20, 5))
        data_title = 'Time Series [ID: {0}]'.format(ts_data_y[i])
        fig.suptitle(data_title, fontsize=12)

        for j in range(n_parts):
            if j == n_parts - 1:
                # Oven temperature
                data_series = ts_data_x[i, oven_cols_index:oven_cols_index + 1, :].flatten()
                axes[j].set_title('Oven Temperature')
            else:
                data_series = ts_data_x[i, j:j + 1, :].flatten()
                axes[j].set_title('({0},{1})'.format(i, j))

            axes[j].get_xaxis().set_visible(False)
            axes[j].plot(data_series)

    plt.show()


from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks as findPeaks


def alsbase(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


def corr_baseline(y):
    y_base = alsbase(y, 10 ^ 5, 0.000005, niter=50)
    corr_y = y - y_base
    return corr_y


def find_series_peaks(ts_data_x, ts_data_y, index_start, length, prominance, distance):
    oven_cols_index = ts_data_x.shape[1] - 1
    n_parts = 5
    for i in range(index_start, index_start + length):
        plt.figure()
        fig, axes = plt.subplots(2, n_parts, figsize=(20, 6))
        data_title = 'Time Series [ID: {0}]'.format(ts_data_y[i])
        fig.suptitle(data_title, fontsize=20)

        for j in range(n_parts):
            if j == n_parts - 1:
                # Oven temperature
                data_series = ts_data_x[i, oven_cols_index:oven_cols_index + 1, :].flatten()
                axes[0, j].set_title('Oven Temperature')
            else:
                data_series = ts_data_x[i, j:j + 1, :].flatten()
                axes[0, j].set_title('({0},{1})'.format(i, j))

            axes[0, j].get_xaxis().set_visible(False)
            axes[0, j].plot(data_series)

        for j in range(n_parts):
            if j == n_parts - 1:
                # Oven temperature
                data_series = ts_data_x[i, oven_cols_index:oven_cols_index + 1, :].flatten()
                axes[1, j].set_title('Oven Temperature')
            else:
                data_series = ts_data_x[i, j:j + 1, :].flatten()
                axes[1, j].set_title('({0},{1})'.format(i, j))

            axes[1, j].get_xaxis().set_visible(False)

            corr_n = corr_baseline(data_series)
            # Locate peaks
            (pks_n, _) = findPeaks(corr_n, prominence=prominance, distance=distance)
            axes[1, j].plot(corr_n)
            axes[1, j].plot(pks_n, corr_n[pks_n], 'x')
            # Locate trough
            (pks_n, _) = findPeaks(corr_n * (-1), prominence=prominance, distance=distance)
            # axes[j].plot(data_series)
            axes[1, j].plot(pks_n, corr_n[pks_n], 'x')
        plt.show()


def init_distance(x_series, y_series):
    dists = np.zeros((len(y_series), len(x_series)))
    for i in range(len(y_series)):
        for j in range(len(x_series)):
            dists[i, j] = (y_series[i] - x_series[j]) ** 2

    return dists


def compute_acu_cost(dists):
    acuCost = np.zeros(dists.shape)
    acuCost[0, 0] = dists[0, 0]

    for j in range(1, dists.shape[1]):
        acuCost[0, j] = dists[0, j] + acuCost[0, j - 1]

    for i in range(1, dists.shape[0]):
        acuCost[i, 0] = dists[i, 0] + acuCost[i - 1, 0]

    for i in range(1, dists.shape[0]):
        for j in range(1, dists.shape[1]):
            acuCost[i, j] = min(acuCost[i - 1, j - 1],
                                acuCost[i - 1, j],
                                acuCost[i, j - 1]) + dists[i, j]

    return acuCost


def compute_dtw_path(x_series, y_series, dists, acuCost):
    i = len(y_series) - 1
    j = len(x_series) - 1

    path = [[j, i]]

    while (i > 0) and (j > 0):
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            if acuCost[i - 1, j] == min(acuCost[i - 1, j - 1],
                                        acuCost[i - 1, j],
                                        acuCost[i, j - 1]):
                i = i - 1
            elif acuCost[i, j - 1] == min(acuCost[i - 1, j - 1],
                                          acuCost[i - 1, j],
                                          acuCost[i, j - 1]):
                j = j - 1
            else:
                i = i - 1
                j = j - 1

        path.append([j, i])

    path.append([0, 0])

    return path


def plt_warp(s1, s2, path, title, xlab="idx", ylab="Value"):
    # plot_fig = plt.figure(figsize=(4, 4))
    plt.title(title, fontsize=10)

    for [idx1, idx2] in path:
        plt.plot([idx1, idx2], [s1[idx1], s2[idx2]],
                 color="C4",
                 linewidth=2)
        plt.plot(s1,
                 'o-',
                 color="C0",
                 markersize=3)
        plt.plot(s2,
                 's-',
                 color="C1",
                 markersize=2)
        plt.xlabel(xlab)
        plt.ylabel(ylab)

    return plt


def perform_dtw(x_series, y_series, title):
    # Initialize distances
    xy_distances = init_distance(x_series, y_series)
    # pltDistances(xy_distances)

    # Compute accumulative cost
    xy_acu_cost = compute_acu_cost(xy_distances)
    # pltDistances(xy_acu_cost, clrmap='Reds')

    # Compute DTW path
    xy_dtw_path = compute_dtw_path(x_series, y_series,
                                   xy_distances, xy_acu_cost)

    # Draw accumulative cost and path
    # cost_path_fig = plt_cost_and_path(xy_acu_cost, xy_dtw_path, title,
    #                                  clrmap='Reds')

    # plt.show()

    # Draw warp on path
    wrap_fit = plt_warp(x_series, y_series, xy_dtw_path, title,
                        xlab="", ylab="")

    # plt.show()


# Perform DTW on OvenTemp and Chip TimeSeries
def perform_dtw_ts(ts_data_x, ts_data_y, ts_data_index_tup, chip_pos_index_tup):
    ts_data_index1, ts_data_index2 = ts_data_index_tup
    chip_pos_index1, chip_pos_index2 = chip_pos_index_tup

    chip_ts1 = ts_data_x[ts_data_index1, chip_pos_index1, :].flatten()
    chip_ts2 = ts_data_x[ts_data_index2, chip_pos_index2, :].flatten()

    dtw_title = '{0}-{1},{2}-{3}'.format(ts_data_y[ts_data_index1], chip_pos_index1,
                                         ts_data_y[ts_data_index2], chip_pos_index2)
    perform_dtw(chip_ts1, chip_ts2, dtw_title)


def perform_dtw_ts_correction(ts_data_x, ts_data_y, ts_data_index_tup, chip_pos_index_tup):
    ts_data_index1, ts_data_index2 = ts_data_index_tup
    chip_pos_index1, chip_pos_index2 = chip_pos_index_tup

    chip_ts1 = ts_data_x[ts_data_index1, chip_pos_index1, :].flatten()
    chip_ts2 = ts_data_x[ts_data_index2, chip_pos_index2, :].flatten()

    chip_corr1 = corr_baseline(chip_ts1)
    chip_corr2 = corr_baseline(chip_ts2)

    dtw_title = '{0}-{1},{2}-{3} (Corr)'.format(ts_data_y[ts_data_index1], chip_pos_index1,
                                                ts_data_y[ts_data_index2], chip_pos_index2)
    perform_dtw(chip_corr1, chip_corr2, dtw_title)


def perform_dtw_comparison_by_chip_pos(ts_data_x, ts_data_y, ts_data_index_tup, chip_pos_index_list):
    chip_pos_index_len = len(chip_pos_index_list)
    rows_plot = int(chip_pos_index_len / 3) + 1
    if chip_pos_index_len % 3 == 0:
        rows_plot -= 1

    data_index = 0
    while data_index < chip_pos_index_len:
        cols_plot = 6
        if chip_pos_index_len - data_index < 3:
            cols_plot = (chip_pos_index_len - data_index) * 2

        plt.figure(figsize=(20, 4))

        for i in range(0, cols_plot, 2):
            subplot_num = int('{0}{1}{2}'.format(1, cols_plot, i + 1))

            axes = plt.subplot(subplot_num)
            perform_dtw_ts(ts_data_x, ts_data_y, ts_data_index_tup, chip_pos_index_list[data_index])
            axes.set_xticks([])
            axes.set_yticks([])

            subplot_num = int('{0}{1}{2}'.format(1, cols_plot, i + 2))
            axes = plt.subplot(subplot_num)
            perform_dtw_ts_correction(ts_data_x, ts_data_y, ts_data_index_tup, chip_pos_index_list[data_index])
            axes.set_xticks([])
            axes.set_yticks([])

            data_index += 1

        plt.show()


def perform_dtw_comparison_by_board_id(ts_data_x, ts_data_y, ts_data_index_list, chip_pos_index_list):
    for ts_index_tup in ts_data_index_list:
        perform_dtw_comparison_by_chip_pos(ts_data_x, ts_data_y, ts_index_tup, chip_pos_index_list)


def plt_burn_in_board(ts_data_x, ts_data_y, ts_data_index_list, clrmap="viridis"):
    # Show board in x * 8 grid
    cols_num = 0
    rows_num = 0

    data_index_len = len(ts_data_index_list)
    total_rows = int(data_index_len / 8) + 1
    if data_index_len % 8 == 0:
        total_rows = total_rows - 1
    if total_rows == 0:
        total_rows = 1

    if total_rows == 1:
        fig, axes = plt.subplots(1, data_index_len, figsize=(20, 6))
    else:
        fig, axes = plt.subplots(total_rows, 8, figsize=(20, 6))

    index_num = 0

    for data_index in ts_data_index_list:
        oven_data_plt = index_num % 2 == 1

        if total_rows == 1:
            if data_index_len == 1:
                plt_axes = axes
            else:
                plt_axes = axes[cols_num]
        else:
            plt_axes = axes[rows_num, cols_num]

        # print('subplot_num: ({0}, {1})'.format(rows_num, cols_num))

        cols_num += 1
        if cols_num >= 8:
            cols_num = 0
            rows_num += 1

        # Exclude last oven temp data
        ts_chip_x = ts_data_x[data_index, 0:640, :].reshape(640, 20)
        ts_chip_mean = np.mean(ts_chip_x, axis=1).reshape(32, 20)

        # ts_oven_x = ts_data_x[data_index, 640, :].reshape(1, 20)
        # ts_oven_mean = np.mean(ts_oven_x, axis=1)

        # ts_x = np.zeros((32, 21))
        # ts_x[:, :-1] = ts_chip_mean
        # ts_x[:, -1] = [ts_oven_mean] * 32

        title = 'ID {0}'.format(ts_data_y[data_index])

        # imgplt = plt.figure(figsize=(6, 6))
        plt_axes.set_title(title, fontsize=20)
        # plt.grid(b=True, which='major', axis='both', color='blue', linestyle='-', linewidth=1)
        plt_axes.imshow(ts_chip_mean, interpolation='nearest', cmap=clrmap)
        # plt_axes.set_xlabel(xlab)
        # plt_axes.set_ylabel(ylab)
        # Remove x and y axis ticks
        plt_axes.set_xticks([0, 5, 10, 15, 19])
        plt_axes.set_yticks([0, 5, 10, 15, 20, 25, 31])

        index_num += 1

    if total_rows > 1:
        while cols_num < 8 and rows_num < total_rows:
            fig.delaxes(axes[rows_num][cols_num])
            cols_num += 1
