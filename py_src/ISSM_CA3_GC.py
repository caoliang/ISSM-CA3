#!/usr/bin/env python
# coding: utf-8

# # Abnormal detection of Chip Temperature Time Series

from numpy.random import seed
seed(37)

from tensorflow import set_random_seed
set_random_seed(29)

from py_src.data_read import *
from py_src.auto_encoder import *
from py_src.signal_functions import *

'''
# Step 6. Check temperature time series
draw_time_series(x_train, Y_train, 0, 1)

# Step 7. Perform Baseline Correction and Locate peaks and trough
find_series_peaks(x_train, Y_train, 0, 1, 0.6, 2)
find_series_peaks(x_train, Y_train, 99, 1, 0.6, 2)
find_series_peaks(x_train, Y_train, 199, 1, 0.6, 2)

# Step 8. Perform DTW (Dynamic Time Wrapping)

# Draw DTW with and without baseline to compare
# 8.1. Perform DTW regarding oven temperature time series
perform_dtw_comparison_by_board_id(x_train, Y_train, [(0, 99)], [(99, 640), (299, 640), (499, 640)])
# 8.2. Perform DTW regarding other time series at same position
perform_dtw_comparison_by_board_id(x_train, Y_train, [(0, 99)], [(99, 99), (299, 299), (499, 499)])

# 8.3. Perform DTW regarding oven temperature time series
perform_dtw_comparison_by_board_id(x_train, Y_train, [(199, 299)], [(99, 640), (299, 640), (499, 640)])
# 8.4. Perform DTW regarding other time series at same position
perform_dtw_comparison_by_board_id(x_train, Y_train, [(199, 299)], [(99, 99), (299, 299), (499, 499)])

# Step 9. Plot temperature differences on the burning

plt.figure()
plt_burn_in_board(x_train, Y_train, range(0, 320, 80), clrmap='gist_heat')
plt.show()
'''

# # 2. Build Autoencoder model
from py_src.auto_encoder import *

# Step 1. Configuration on autoencoder model

# Step 2. Define an autoencoder model

# Build an autoencoder model

# Remove oven temperature time series
# ae_data_train = x_train[:, 420:421]
# ae_data_test = x_test[:, 420:421]

def get_threshold(x_data, y_data, model_AE, data_columns, threshold_percentage):
    data_cols = data_columns[0:x_data.shape[1]]

    train_pred = model_AE.predict(x_data)
    X_pred = pd.DataFrame(train_pred, columns=data_cols)
    X_pred.index = y_data

    sqrt_n = np.sqrt(x_data.shape[1])
    tempSqrtValue = rmse_sqrt_n(x_data, X_pred, sqrt_n)
    index_position = int(len(tempSqrtValue) * threshold_percentage / 100)
    print("index_position", index_position)
    return sorted(tempSqrtValue, reverse=True)[index_position]

def get_threshold_new(x_data, train_pred, threshold_percentage):
    x_pred = pd.DataFrame(train_pred)

    print("get_threshold_new--------->x_data.shape[1]:", x_data.shape[1])
    sqrt_n = np.sqrt(x_data.shape[1])

    tempSqrtValue = rmse_sqrt_n(x_data, x_pred, sqrt_n)

    index_position = int(len(tempSqrtValue) * threshold_percentage / 100)

    print("index_position", index_position)

    return sorted(tempSqrtValue, reverse=True)[index_position]

def show_rmse_hist(train_x_data, train_Y_data, model_AE, data_columns, hist_bins=30, hist_title='LOSS RMAE'):
    plt.figure()
    data_cols = data_columns[0:train_x_data.shape[1]]
    train_pred = model_AE.predict(train_x_data)

    X_pred = pd.DataFrame(train_pred, columns=data_cols)
    X_pred.index = train_Y_data

    scored = pd.DataFrame(index=train_Y_data)
    sqrt_n = np.sqrt(train_x_data.shape[1])

    scored[hist_title] = rmse_sqrt_n(train_x_data, X_pred, sqrt_n)
    scored.hist(bins=hist_bins)
    plt.show()

def plot_original_img(ae_data_train):
    plt.figure()
    fig, ax = plt.subplots(1, 10)
    for i in range(10):
        ax[i].imshow(ae_data_train[i], cmap='gray')
        ax[i].axis('off')
    plt.show()


def process_single_chip(ae_data_train, ae_data_test, threshold_percentage):
    print("ae_data_train.shape:", ae_data_train.shape, "ae_data_test.shape:", ae_data_test.shape)

    ae_data_max = np.amax(ae_data_train)

    # Normalize
    ae_data_train = ae_data_train.astype('float32') * 255 / ae_data_max
    ae_data_test = ae_data_test.astype('float32') * 255 / ae_data_max

    # Plot the examples of original images
    # plot_original_img(ae_data_train)

    ae_data_train = ae_data_train.reshape((ae_data_train.shape[0], -1))
    ae_data_test = ae_data_test.reshape((ae_data_test.shape[0], -1))
    print("ae_data_train.shape:", ae_data_train.shape, "ae_data_test.shape:", ae_data_test.shape)

    # Train the model
    model_encoder, model_auto_encoder = build_AE(x_train=ae_data_train)

    # Visualize the results
    num_sample = 3  # Number of test samples used for visualization
    # ae_data_pred = model_auto_encoder.predict(ae_data_test)
    # show_encodings(ae_data_test[:num_sample], ae_data_pred)

    #threshold = get_threshold(ae_data_train, Y_train, model_auto_encoder, cols_name_list, threshold_percentage)
    train_pred = model_auto_encoder.predict(ae_data_train)
    threshold = get_threshold_new(ae_data_train, train_pred, threshold_percentage)

    print("threshold:", threshold)

    # show_rmse_hist(ae_data_train, Y_train, model_autoencoder, cols_name_list, hist_bins=50, hist_title='Training Data Loss RMSE')

    # Step 2: Perform prediction using the trained autoencoder model
    test_data_predict = model_auto_encoder.predict(ae_data_test)

    # Calculate the reconstruction error and make decision on anomaly detection
    check_anomaly_and_compute_dist(ae_data_test, test_data_predict, threshold=threshold)

# last one is oven temperature, need to remove it ???
total_length = len(x_train[0]) - 1
threshold_percentage = 5

for curr_idx in range(0, total_length):
    ae_data_train = x_train[:, curr_idx : curr_idx+block_chips]
    ae_data_test = x_test[:, curr_idx : curr_idx+block_chips]
    process_single_chip(ae_data_train, ae_data_test, threshold_percentage)