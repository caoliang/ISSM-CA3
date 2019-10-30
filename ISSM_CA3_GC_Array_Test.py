import pandas as pd
import numpy as np
import h5py
import os
import time
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from keras import regularizers
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import UpSampling3D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool3D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_AE(optimizer=None, signal_dim=20, encoder_dim=5):
    # input placeholder
    input_signal = Input(shape=(signal_dim, ))

    # encoding layer
    hidden_layer = Dense(encoder_dim, activation='relu', 
                         kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(0.0))(input_signal)

    # decoding layer
    decode_output = Dense(signal_dim, activation='relu',
                          kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(0.0))(hidden_layer)

    # build autoencoder
    autoencoder = Model(inputs=input_signal, outputs=decode_output)

    # compile autoencoder
    autoencoder.compile(optimizer=optimizer, loss='mse')

    return autoencoder

def build_AE_array(rows, cols, signal_len=20, encoder_len=16, lr_rate=0.0001,
                   lr_scheduler=None):
    autoencoder_list = []
    
    for i in range(rows):
        for j in range(cols):
            if i % 10 == 0 and j == 0:
                print('Build AE array [{0}, {1}]'.format(i, j))
            optmz = optimizers.Adam(lr=learning_rate)
            
            autoencoder = build_AE(optimizer=optmz, signal_dim=signal_len,
                                   encoder_dim=encoder_len)
            autoencoder_list.append(autoencoder)
    
    autoencoder_array = np.array(autoencoder_list).reshape(rows, cols)
    print('Complete to build AE array: ', autoencoder_array.shape)
    return autoencoder_array

def get_AE_model_file(loc_row, loc_col, model_folder):
    model_filename = 'AE_array_{0}-{1}.hdf5'.format(loc_row, loc_col)
    model_file = os.path.join(model_folder, model_filename)
    
    return model_file

def get_AE_epoch_file(loc_row, loc_col, epoch_folder):
    epoch_filename = 'AE_array_train_{0}-{1}.csv'.format(loc_row, loc_col)
    epoch_file = os.path.join(epoch_folder, epoch_filename)
    
    return epoch_file

def predict_data(ae_model_array, data_x):
    data_size = data_x.shape[0]
    rows = data_x.shape[1]
    cols = data_x.shape[2]
    signal_size = data_x.shape[3]
    
    data_pred = np.copy(data_x)
    
    for i in range(rows):
        for j in range(cols):
            timer_check_j = time.time()
            print('Predict with AE array [{0}, {1}]'.format(i, j))
            data_x_i_j = data_x[:, i, j, :, :].reshape(data_size, signal_size)
            pred_x_i_j = ae_model_array[i, j].predict(data_x_i_j)
            data_pred[:, i, j, :, 0] = pred_x_i_j
            
            print('Complete prediction with AE array [{0}, {1}] in {2}'.format(
                i, j, show_time_spent(timer_check_j)))            

    print('Complete to predict with AE array as : ', data_pred.shape)
    return data_pred


def load_AE(model_weights_file=None, optimizer=None, 
            signal_size=20, encoder_size=16):
    ae_model = build_AE(optimizer=optimizer, signal_dim=signal_size, 
                        encoder_dim=encoder_size)
    ae_model.load_weights(model_weights_file)
    
    return ae_model

def show_time_spent(start_time):
    time_used = time.time() - start_time
    return "{}:{}:{}".format(int(time_used / 3600), int(time_used % 3600 / 60), int(time_used % 60))

def load_AE_array(rows, cols, model_weights_folder=None,
                  signal_size=20, encoder_size=16):

    optmz = optimizers.Adam(lr=0.0001)
    timer_start = time.time()
    ae_model_list = []
    for i in range(rows):
        timer_check = time.time()

        for j in range(cols):
            timer_check_j = time.time()
            print('Load AE array: [{0}, {1}]'.format(i, j))
            ae_model_file = get_AE_model_file(i, j, model_weights_folder)
            ae_model = load_AE(model_weights_file=ae_model_file, 
                               optimizer=optmz, signal_size=signal_size, 
                               encoder_size=encoder_size)
            ae_model_list.append(ae_model)
            print('Complete load AE array [{0}, {1}] in {2}'.format(
                i, j, show_time_spent(timer_check_j)))            

        print('Complete load AE array [{0}] in {1}'.format(
                i, show_time_spent(timer_check)))            

    ae_model_array = np.array(ae_model_list).reshape(rows, cols)
    print('Completed to load AE array ({0}) in {1}'.format(
            ae_model_array.shape, show_time_spent(timer_start)))
            
    return ae_model_array

def show_time_series_image(inputs_x, outputs_x, loc_index, 
                           images_in_row=4, clrmap='gray'):
    img_row_size = int(inputs_x.shape[0] / images_in_row)
    #print('img_row_size: ', img_row_size)
    
    inputs_x_data = inputs_x.reshape(inputs_x.shape[0], 
                                     inputs_x.shape[1] * inputs_x.shape[2],
                                     20)

    outputs_x_data = outputs_x.reshape(outputs_x.shape[0], 
                                       outputs_x.shape[1] * outputs_x.shape[2],
                                       20)
    
    #print('inputs_x_data, shape: ', inputs_x_data.shape)
    #print('outputs_x_data, shape: ', outputs_x_data.shape)
        
    fig, axes = plt.subplots(2, images_in_row, figsize=(20,10))
    for ax in axes.flatten():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    for i in range(images_in_row):
        inputs_img = inputs_x_data[i * img_row_size:(i+1) * img_row_size, 
                                   loc_index, :].reshape(img_row_size, 20)
        outputs_img = outputs_x_data[i * img_row_size:(i+1) * img_row_size,
                                     loc_index, :].reshape(img_row_size, 20)
        
        axes[0, i].imshow(inputs_img, cmap=clrmap, aspect='auto')
        axes[1, i].imshow(outputs_img, cmap=clrmap, aspect='auto')

    plt.show()

def show_board_snapshot(data_x, data_y, x_index_list, 
                        clrmap="viridis"):
    #plt.figure()
        
    # Show board in x * 8 grid
    cols_num = 0
    rows_num = 0
    
    data_index_len = len(x_index_list)
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
    
    for data_index in x_index_list:
        if total_rows == 1:
            if data_index_len == 1:
                plt_axes = axes
            else:
                plt_axes = axes[cols_num]
        else:
            plt_axes = axes[rows_num, cols_num]
            
        #print('subplot_num: ({0}, {1})'.format(rows_num, cols_num))
                
        cols_num += 1
        if cols_num >= 8:
            cols_num = 0
            rows_num += 1
    
        
        # Show mean value in each chip temperature time series
        ts_chip_x = data_x[data_index, :, :]
        ts_chip_mean = np.mean(ts_chip_x, axis=1).reshape(32, 20)
        
        title = 'ID {0}'.format(data_y[data_index])

        #imgplt = plt.figure(figsize=(6, 6))
        plt_axes.set_title(title, fontsize=20)
        #plt.grid(b=True, which='major', axis='both', color='blue', linestyle='-', linewidth=1)
        plt_axes.imshow(ts_chip_mean, interpolation='nearest', cmap=clrmap)
        #plt_axes.set_xlabel(xlab)
        #plt_axes.set_ylabel(ylab)
        # Remove x and y axis ticks
        plt_axes.set_xticks([0, 5, 10, 15, 19])
        plt_axes.set_yticks([0, 5, 10, 15, 20, 25, 31])
        
        index_num += 1

    if total_rows > 1:
        while cols_num < 8 and rows_num < total_rows:
            fig.delaxes(axes[rows_num][cols_num])
            cols_num += 1

    plt.show()


def show_mse_hist(data_x, encoded_x, data_threshold, loc_row, loc_col,
                  hist_bins=100, hist_title='MSE Histogram'):
    board_size = data_x.shape[0]
    row_size = data_x.shape[1]
    col_size = data_x.shape[2]
    ts_size = data_x.shape[3]
    
#    print('MSE Histogram [{0}, {1}], Threshold: {2}'.format(loc_row, loc_col,
#          data_threshold[loc_row, loc_col]))
    
    scored = pd.DataFrame()
    chip_data = data_x[:, loc_row, loc_col, :, :].reshape(board_size, ts_size)
    chip_encoded = encoded_x[:, loc_row, loc_col, :, :].reshape(board_size, ts_size)
    
    chip_hist_title = '{0} [{1}, {2}] Threshold {3:.4f}'.format(
            hist_title, loc_row, loc_col, data_threshold[loc_row, loc_col])
    
    scored[chip_hist_title] = compute_mse(chip_data, chip_encoded)
    hist = scored.hist(bins = hist_bins)

def draw_time_series(data_ts, data_y, loc_row, loc_col, 
                     x_index_list, prefix_title=''):
    n_parts = len(x_index_list)
    fig, axes = plt.subplots(1, n_parts, figsize=(n_parts * 5, 5))
    axes_index = 0    
    
    for data_index in x_index_list:
        data_title = '{0}{1} [{2},{3}]'.format(prefix_title,
                      data_y[data_index], loc_row, loc_col)
        data_series = data_ts[data_index, :].flatten()
        if n_parts == 1:
            axes_n = axes
        else:
            axes_n = axes[axes_index]
        axes_n.set_title(data_title)
        
        axes_n.get_xaxis().set_visible(False)
        axes_n.plot(data_series)
        axes_index += 1        

    plt.show()


def show_abnormal_status(data_x, data_y, check_abnormal_status, 
                         loc_row, loc_col):
    board_size = data_x.shape[0]
    row_size = data_x.shape[1]
    col_size = data_x.shape[2]
    ts_size = data_x.shape[3]
    
    data_ts_x = data_x[:, loc_row, loc_col, :, :].reshape(board_size, ts_size)
    check_status = check_abnormal_status[:, loc_row, loc_col].reshape(board_size)
    abnormal_status = []
    normal_status = []
    
    for i in range(board_size):
        if check_status[i]:
            abnormal_status.append(i)
        else:
            normal_status.append(i)
    
    if len(abnormal_status) == 0:
        print('Abnormal chip not found at [{0},{1}]'.format(loc_row, loc_col))
    else:
        # Only show 4 sample
        abnormal_status = abnormal_status[0:4]
        draw_time_series(data_ts_x, data_y, loc_row, loc_col, 
                         abnormal_status, prefix_title='Abnormal - ')
    
    if len(normal_status) == 0:
        print('Normal chip not found at [{0},{1}]'.format(loc_row, loc_col))
    else:
        # Only show 4 sample
        normal_status = normal_status[0:4]
        draw_time_series(data_ts_x, data_y, loc_row, loc_col, 
                         normal_status, prefix_title='Normal - ')


def check_abnormal(data_x, encoded_x, threshold_data):
    board_size = data_x.shape[0]
    row_size = data_x.shape[1]
    col_size = data_x.shape[2]
    ts_size = data_x.shape[3]
    status_list = np.zeros((board_size, row_size, col_size))

    for i in range(row_size):
        for j in range(col_size):
            loc_index = i * row_size + j
            threshold_ts = threshold_data[i, j]
            data_ts = data_x[:, i, j, :, :].reshape(board_size, ts_size)
            encoded_ts = encoded_x[:, i, j, :, :].reshape(board_size, ts_size)
            mse_ts = compute_mse(data_ts, encoded_ts)
            status_ts = mse_ts >= threshold_ts
            status_list[:, i, j] = status_ts

    return status_list


def compute_mse(data_in, data_pred):
    return np.linalg.norm(data_pred - data_in, axis=-1)

def compute_mse_threshold(data_x, encoded_x, top_percentage):
    board_size = data_x.shape[0]
    row_size = data_x.shape[1]
    col_size = data_x.shape[2]
    ts_size = data_x.shape[3]
    loc_index = 0
    threshold_list = []
    
    for i in range(row_size):
        for j in range(col_size):
            loc_index = i * row_size + j
            data_ts = data_x[:, i, j, :, :].reshape(board_size, ts_size)
            encoded_ts = encoded_x[:, i, j, :, :].reshape(board_size, ts_size)
            mse_ts = compute_mse(data_ts, encoded_ts)
            mse_hist, mse_hist_edges = np.histogram(mse_ts, bins=100)
            mse_threshold = mse_hist_edges[-top_percentage]
            
            threshold_list.append(mse_threshold)
        
#            if loc_index % 100 == 0:
#                print('[{0},{1}] = {2:.5f}'.format(i, j, mse_threshold))

    threshold_arr = np.array(threshold_list).reshape(row_size, col_size)
    print(threshold_arr.shape)

    return threshold_arr

def read_mse_threshold(h5_in_file='mse_threshold.h5'):
    with h5py.File(h5_in_file, 'r') as hf:
        mse_threshold = hf['mse_threshold'].value
        print('Read mse_threshold: ', mse_threshold.shape)        
        
    return mse_threshold

def read_train_test_dataset(h5_in_file='dataset.h5'):
    with h5py.File(h5_in_file, 'r') as hf:
        train_data = hf['train_data'].value
        print('Read train_data: ', train_data.shape)
        
        train_lbl = hf['train_lbl'].value
        print('Read train_lbl: ', train_lbl.shape)
        
        test_data = hf['test_data'].value
        print('Read test_data: ', test_data.shape)
        
        test_lbl = hf['test_lbl'].value
        print('Read test_lbl: ', test_lbl.shape)
                
    return (train_data, train_lbl, test_data, test_lbl)

    # fix random seed for reproducibility
seed = 49
np.random.seed(seed)

plt.style.use('ggplot')
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False

modelname = 'CA3ModelV1'
saved_dataset_file = modelname + '_dataset.hdf5'
saved_model_folder = modelname + '_model_array_trained'
chips_temp_file = 'data/temp_data.csv'
saved_threshold_file = modelname + '_model_array_threshold_trained.hdf5'

ts_feature_units = 10
mse_threshold_percent = 1

learning_rate = 1e-4
epochs_size = 40
batch_size = 10

    # Whether show model summary
show_model_summary = True

    # Whether show prediction differences on board
show_board_diff = True

    # Show individual chip location time series differences
show_chip_diff = True

    # Show individula chip location threshold differences
show_threshold_diff = True

tr_data, tr_lbl, ts_data, ts_lbl = read_train_test_dataset(
        h5_in_file=saved_dataset_file)
    
    # Convert the data into 'float32'
    # Rescale the values from 0~255 to 0~1
tr_data = tr_data.astype('float32') / 255
ts_data = ts_data.astype('float32') / 255

    # Retrieve the row, column, depth, channel size of each time series
tsrows = tr_data.shape[1]
tsclms = tr_data.shape[2]
tslen = tr_data.shape[3]
channel = tr_data.shape[4]

check_pos_list = [(0, 0), (0, 9), (0,19),
               (7, 0), (7, 9), (7,19),
               (15,0), (15,9), (15,19),
               (23,0), (23,9), (23,19),
               (31,0), (31,9), (31,19)]

    # Build autoencoder
pretrained_model_array =  load_AE_array(tsrows, tsclms, 
                                        model_weights_folder=saved_model_folder,
                                        signal_size=tslen, 
                                        encoder_size=ts_feature_units)

if show_model_summary:
    print()
    print('pretrained autoencoder model')
    pretrained_model_array[0, 0].summary()

    # Predict testing data
encoded_ts_data = predict_data(pretrained_model_array, ts_data)

if show_board_diff:
    print()
    print('Test data, shape: ', ts_data.shape)
    print('Decoded Test data, shape: ', encoded_ts_data.shape)

    print()
    print('Test Data - Borad Temperature ')
    show_board_snapshot(ts_data, ts_lbl, 
                        range(0, ts_data.shape[0], 
                              int(ts_data.shape[0] / 4)), 
                        clrmap='gist_heat')
        # Show decoded data
    print()
    print('Encoded Test Data - Borad Temperature ')
    show_board_snapshot(encoded_ts_data, ts_lbl, 
                        range(0, encoded_ts_data.shape[0], 
                              int(encoded_ts_data.shape[0] / 5)), 
                        clrmap='gist_heat')

if show_chip_diff:
    for chip_loc_x, chip_loc_y in check_pos_list:
        chip_loc = chip_loc_x * 20 + chip_loc_y
        
        print()
        print(('Training vs Encoded Data - Chip [{0}, {1}] Temperature '
               ).format(chip_loc_x, chip_loc_y))
        show_time_series_image(ts_data, encoded_ts_data, chip_loc,
                               images_in_row=8, clrmap='gist_heat')
        

    # Read threshold trained
threshold_saved = read_mse_threshold(h5_in_file=saved_threshold_file)

    # Check abnormal status
check_status = check_abnormal(ts_data, encoded_ts_data, threshold_saved)
        
if show_threshold_diff:
    for pos in check_pos_list:
        show_mse_hist(ts_data, encoded_ts_data, threshold_saved,
                  pos[0], pos[1])
        show_abnormal_status(ts_data, ts_lbl, check_status,
                             pos[0], pos[1])

        