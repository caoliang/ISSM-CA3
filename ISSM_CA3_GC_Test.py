import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
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
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_dataset(number_of_samples=10):
    mean_temp = 80
    var_temp = 0.8
    
    ts_X = np.random.normal(mean_temp, var_temp, 
                            size=(number_of_samples, 32, 20, 20))
    #print('ts_x', ts_X[0:1])            
        
    # 5d shape with channels last
    ts_X = np.array(ts_X).reshape((number_of_samples, 32, 20, 20, 1))
    ts_Y = np.array(range(number_of_samples))
    
    size_train = int(number_of_samples / 10 * 8)
    
    X_train = ts_X[0:size_train, :, :, :]
    print('Read X_train: ', X_train.shape)
    
    y_train = ts_Y[0:size_train]
    print('Read y_train: ', y_train.shape)        
    
    X_test = ts_X[size_train:number_of_samples, :, :, :]
    print('Read X_test: ', X_test.shape)        
    
    y_test = ts_Y[size_train:number_of_samples]
    print('Read y_test: ', y_test.shape)        

    return (X_train, y_train, X_test, y_test)

def prepare_dataset(data_csv_file='data/temp_data.csv'):
        # Step 1. Read chip temperature data from CSV file
    temp_base_data = pd.read_csv(data_csv_file)
    print(temp_base_data.shape)

        # Step 2. Divide data into time series data based on 
        #         burn-in board coordinates
    temp_data = temp_base_data.pivot(index='ID', columns='Time(min)')
    print(temp_data.shape)
    
        # Rename column name as string value
    data_cols = list(temp_data.columns)
    print(data_cols[0:1])
    cols_name_list = ['{0}_{1}'.format(col_item[0], col_item[1]) for col_item 
                      in data_cols]
    print(cols_name_list[0:1])
    temp_data.columns = cols_name_list
    print(len(list(temp_data.columns)))
    
        # Step 3. Check Null value
    null_value_sum = temp_data.isnull().sum().sum()
    print('Total null value in dataset: ', null_value_sum)

        # Step 4. Divide data to Time Series data and Chip ID data
    temp_data_y = temp_data.index.values
    temp_data_x = temp_data.values.reshape(400, 641, 20)
    
        # Remove oven temperature data
    temp_data_x = temp_data_x[:, 0:-1, :]
    temp_data_x = temp_data_x.reshape(400, 32, 20, 20, 1)
    
    print('temp_data_x: ', temp_data_x.shape)
    print('temp_data_y: ', temp_data_y.shape)

    x_train, x_test, Y_train, Y_test = train_test_split(temp_data_x, 
                                                        temp_data_y, 
                                                        test_size=0.20, 
                                                        random_state=49)
    print("Training data: ", x_train.shape, ", Training ID: ", Y_train.shape)
    print("Testing data: ", x_test.shape, ", Testing ID: ", Y_test.shape)

    return (x_train, Y_train, x_test, Y_test)
    

def build_3d_AE(optimizer=None, dropout_rate=0.2, num_units=16):
    
    
        # input 5D shape
    input_data = Input(shape=(32, 20, 20, 1))

        # Inner shape before conv3dtranspose
    inner_shape = (16, 10, 20)

        # encoding layer
    encode_layer = Conv3D(16, kernel_size=(2, 2, 2), 
                          strides=(1, 1, 1), 
                          activation='relu',
                          padding='same', 
                          data_format="channels_last")(input_data)

#        # max pooling 3D
#    encode_layer = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
#                             padding='same')(encode_layer)

    encode_layer = Conv3D(32, kernel_size=(2, 2, 2), 
                          strides=(2, 2, 1), 
                          activation='relu',
                          padding='same', 
                          data_format="channels_last")(encode_layer)
        
    encode_layer = Flatten()(encode_layer)
    encode_layer = Dense(2 * num_units, activation='relu')(encode_layer)
    encode_layer = Dense(num_units, activation='relu')(encode_layer)
    
        # decoding layer
    decode_layer = Dense(2 * num_units, activation='relu')(encode_layer)
    decode_layer = Dense(inner_shape[0] * inner_shape[1] * 
                         inner_shape[2] * 32, activation='relu')(decode_layer)
    decode_layer = Reshape((inner_shape[0], inner_shape[1], 
                            inner_shape[2], 32))(decode_layer)
    
    decode_layer = Conv3DTranspose(32, kernel_size=(2, 2, 2), 
                                   strides=(2, 2, 1), 
                                   activation='relu',
                                   padding='same', 
                                   data_format="channels_last")(decode_layer)
#        # upsampling 3D
#    decode_layer = UpSampling3D(size=(2, 2, 2), 
#                                data_format="channels_last")(decode_layer)    
                                                       
    decode_layer = Conv3DTranspose(1, kernel_size=(2, 2, 2), 
                                   strides=(1, 1, 1), 
                                   activation='relu',
                                   padding='same', 
                                   data_format="channels_last")(decode_layer)
        
    # build autoencoder
    autoencoder = Model(inputs=input_data, outputs=decode_layer)

    # compile autoencoder
    autoencoder.compile(optimizer=optimizer, loss='mse',
                        metrics=['mean_squared_error'])

    return autoencoder


def train_3d_AE(ae_model=None, x_train=None, x_test=None, 
                model_out_file=None, epoch_out_file=None, 
                epochs=10, batch_size=10):
    # Create checkpoint for the training
    # This checkpoint performs model saving when
    # an epoch gives highest testing accuracy
    checkpoint = ModelCheckpoint(model_out_file, 
                                  monitor='val_loss', 
                                  verbose=1, 
                                  save_best_only=True, 
                                  mode='min')
    # Log the epoch detail into csv
    csv_logger = CSVLogger(epoch_out_file)
    callbacks_list = [checkpoint, csv_logger]
    
    # Fit the model
    ae_model.fit(x_train, x_train, validation_data=(x_test, x_test), 
                 epochs=epochs, batch_size=batch_size,
                 shuffle=True, callbacks=callbacks_list)


def show_training(epoch_out_file=None):
    records     = pd.read_csv(epoch_out_file)
    plt.figure()
    plt.plot(records['val_loss'])
    plt.plot(records['loss'])
    plt.yticks([0.00,0.05, 0.10,0.15,0.20])
    plt.title('Loss value',fontsize=12)
#    ax = plt.gca()
#    ax.set_xticklabels([])
    

def export_3d_AE(ae_model=None, export_model_file=None):
    from tensorflow.keras.utils import plot_model

    plot_model(ae_model, to_file=export_model_file, 
               show_shapes=True, show_layer_names=False,
               rankdir='TB')    


def load_3d_AE(model_weights_file=None, optimizer=None, 
               dropout_rate=0.2, num_units=16):
    ae_3d_model = build_3d_AE(optimizer=optimizer, dropout_rate=dropout_rate, 
                              num_units=num_units)
    ae_3d_model.load_weights(model_weights_file)
    
    return ae_3d_model


def show_time_series_image(inputs_x, outputs_x, loc_index, 
                           images_in_row=4, clrmap='gray'):
    img_row_size = int(inputs_x.shape[0] / images_in_row)
    print('img_row_size: ', img_row_size)
    
    inputs_x_data = inputs_x.reshape(inputs_x.shape[0], 
                                     inputs_x.shape[1] * inputs_x.shape[2],
                                     20)

    outputs_x_data = outputs_x.reshape(outputs_x.shape[0], 
                                       outputs_x.shape[1] * outputs_x.shape[2],
                                       20)
    
    print('inputs_x_data, shape: ', inputs_x_data.shape)
    print('outputs_x_data, shape: ', outputs_x_data.shape)
        
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
        
        #ts_oven_x = ts_data_x[data_index, 640, :].reshape(1, 20)
        #ts_oven_mean = np.mean(ts_oven_x, axis=1)
        
        #ts_x = np.zeros((32, 21))
        #ts_x[:, :-1] = ts_chip_mean
        #ts_x[:, -1] = [ts_oven_mean] * 32
        
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


    # fix random seed for reproducibility
seed = 49
np.random.seed(seed)

plt.style.use('ggplot')
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False

modelname = 'CA3ModelV1'
saved_model_file = modelname + '_model.hdf5'
saved_training_file = modelname + '_train.csv'
saved_model_design_file = modelname + '_design.pdf'
chips_temp_file = 'data/temp_data.csv'
num_feature_units = 64

optmz = optimizers.Adam(lr=0.00001)

    # Whether perform training
perform_training_process = True

    # Whether show trained model
show_trained_model = True

    # Whether using real data or random generated data (experiment only) 
test_on_real_data = True

if test_on_real_data:
    tr_data, tr_lbl, ts_data, ts_lbl = prepare_dataset(
            data_csv_file=chips_temp_file)
    
else:
        # Generate dataset for testing
    total_num_data = 400
    tr_data, tr_lbl, ts_data, ts_lbl = generate_dataset(total_num_data)

    # Convert the data into 'float32'
    # Rescale the values from 0~255 to 0~1
tr_data = tr_data.astype('float32') / 255
ts_data = ts_data.astype('float32') / 255

    # Retrieve the row, column, depth, channel size of each time series
imgrows = tr_data.shape[1]
imgclms = tr_data.shape[2]
imgdepth = tr_data.shape[3]
channel = tr_data.shape[4]

if perform_training_process:

        # Build autoencoder
    ae_3d_model = build_3d_AE(optimizer=optmz, dropout_rate=0.2, 
                              num_units=num_feature_units)
    ae_3d_model.summary()
    
        # Train autoencoder
    train_3d_AE(ae_3d_model, x_train=tr_data, x_test=ts_data, 
                model_out_file=saved_model_file,
                epoch_out_file=saved_training_file,
                epochs=50, batch_size=32)
    
        # Show training
    show_training(epoch_out_file=saved_training_file)
    
        # Export model design
    export_3d_AE(ae_model=ae_3d_model, 
                 export_model_file=saved_model_design_file)

if show_trained_model:
    
        # Whether show model summary
    show_model_summary = False
    
        # Whether show prediction differences on board
    show_board_diff = False
    
        # Show individual chip location time series differences
    show_chip_diff = True
    
        # Build autoencoder
    pretrained_ae_model = load_3d_AE(model_weights_file=saved_model_file,
                             optimizer=optmz, dropout_rate=0.2, 
                             num_units=num_feature_units)
    if show_model_summary:
        print()
        print('pretrained autoencoder model')
        pretrained_ae_model.summary()

        # Load pre-trained model
    decoded_tr_data = pretrained_ae_model.predict(tr_data)

    if show_board_diff:
            # Show original data
        print()
        print('Original data, shape: ', tr_data.shape)
        show_board_snapshot(tr_data, tr_lbl, 
                            range(0, tr_data.shape[0], 
                                  int(tr_data.shape[0] / 4)), 
                            clrmap='gist_heat')
            # Show decoded data
        print()
        print('Decoded data, shape: ', decoded_tr_data.shape)
        show_board_snapshot(decoded_tr_data, tr_lbl, 
                            range(0, decoded_tr_data.shape[0], 
                                  int(decoded_tr_data.shape[0] / 5)), 
                            clrmap='gist_heat')

    if show_chip_diff:
        chip_loc_x = 0
        chip_loc_y = 0
        chip_loc = chip_loc_x * 20 + chip_loc_y
        
        print()
        print(('Show chip location [{0}, {1}] temperature time series '
              + 'differences').format(chip_loc_x, chip_loc_y))
        
        show_time_series_image(tr_data, decoded_tr_data, chip_loc,
                               images_in_row=8, clrmap='gist_heat')
        