import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
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
    mean = 90
    sigma = 0.8
    
    ts_X = np.random.normal(loc=mean, scale=sigma, 
                            size=(number_of_samples, 32, 20, 20))
    
    # 5d shape with channels last
    ts_X = np.array(ts_X).reshape((number_of_samples, 32, 20, 20, 1))
    ts_Y = np.array(range(number_of_samples))
    
    size_train = int(number_of_samples / 2)
    
    X_train = ts_X[0:size_train, :, :, :]
    print('Read X_train: ', X_train.shape)
    
    y_train = ts_Y[0:size_train]
    print('Read y_train: ', y_train.shape)        
    
    X_test = ts_X[size_train:number_of_samples, :, :, :]
    print('Read X_test: ', X_test.shape)        
    
    y_test = ts_Y[size_train:number_of_samples]
    print('Read y_test: ', y_test.shape)        

    return (X_train, y_train, X_test, y_test)


def build_3d_AE(optimizer=None, dropout_rate=0.2, num_units=16):
    
    
    # input 5D shape
    input_data = Input(shape=(32, 20, 20, 1))

    # encoding layer
    encode_layer = Conv3D(16, kernel_size=(2, 2, 2), 
                          strides=(1, 1, 1), 
                          activation='selu',
                          padding='same', 
                          data_format="channels_last")(input_data)

    # max pooling 3D
#    encode_layer = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
#                             padding='same')(encode_layer)

    encode_layer = Conv3D(32, kernel_size=(2, 2, 2), 
                          strides=(2, 2, 2), 
                          activation='selu',
                          padding='same', 
                          data_format="channels_last")(encode_layer)
    
    
    encode_layer = Flatten()(encode_layer)
    encode_layer = Dense(2 * num_units, activation='relu')(encode_layer)
    encode_layer = Dense(num_units, activation='relu')(encode_layer)
    
    # decoding layer
    decode_layer = Dense(2 * num_units, activation='relu')(encode_layer)
    decode_layer = Dense(16 * 10 * 10 * 32, activation='relu')(decode_layer)
    decode_layer = Reshape((16, 10, 10, 32))(decode_layer)
    
    decode_layer = Conv3DTranspose(32, kernel_size=(2, 2, 2), 
                                   strides=(2, 2, 2), 
                                   activation='selu',
                                   padding='same', 
                                   data_format="channels_last")(decode_layer)
                                                              
    decode_layer = Conv3DTranspose(1, kernel_size=(2, 2, 2), 
                                   strides=(1, 1, 1), 
                                   activation='selu',
                                   padding='same', 
                                   data_format="channels_last")(decode_layer)
        
    # build autoencoder
    # encoder = Model(inputs=input_data, outputs=encode_layer)
    autoencoder = Model(inputs=input_data, outputs=decode_layer)

    # compile autoencoder
    autoencoder.compile(optimizer=optimizer, loss='mse',
                        metrics=['accuracy'])

    return autoencoder

def train_3d_AE(ae_model=None, x_train=None, x_test=None, 
                model_out_file=None, epoch_out_file=None, 
                epochs=10, batch_size=10):
    # Create checkpoint for the training
    # This checkpoint performs model saving when
    # an epoch gives highest testing accuracy
    checkpoint = ModelCheckpoint(model_out_file, 
                                  monitor='val_acc', 
                                  verbose=1, 
                                  save_best_only=True, 
                                  mode='max')
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
    plt.subplot(211)
    plt.plot(records['val_loss'])
    plt.plot(records['loss'])
    plt.yticks([0.00,0.40,0.60,0.80])
    plt.title('Loss value',fontsize=12)
    
    ax = plt.gca()
    ax.set_xticklabels([])
    
    plt.subplot(212)
    plt.plot(records['val_acc'])
    plt.plot(records['acc'])
    plt.yticks([0.6,0.7,0.8,0.9])
    plt.title('Accuracy',fontsize=12)
    plt.show()

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


# fix random seed for reproducibility
seed = 49
np.random.seed(seed)

plt.style.use('ggplot')
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False

total_num_data = 20

# Generate dataset for testing
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

optmz = optimizers.Adam(lr=0.001)
modelname = 'CA3ModelV1'
saved_model_file = modelname + '_model.hdf5'
saved_training_file = modelname + '_train.csv'
saved_model_design_file = modelname + '_design.pdf'

# Build autoencoder
ae_3d_model = build_3d_AE(optimizer=optmz, dropout_rate=0.2, num_units=16)
ae_3d_model.summary()

# Train autoencoder
train_3d_AE(ae_3d_model, x_train=tr_data, x_test=ts_data, 
            model_out_file=saved_model_file,
            epoch_out_file=saved_training_file,
            epochs=10, batch_size=2)

# Show training
show_training(epoch_out_file=saved_training_file)

# Export model design
export_3d_AE(ae_model=ae_3d_model, 
             export_model_file=saved_model_design_file)

# Build autoencoder
pretrained_ae_model = load_3d_AE(model_weights_file=saved_model_file,
                         optimizer=optmz, dropout_rate=0.2, num_units=16)
print()
print('pretrained autoencode model')
pretrained_ae_model.summary()

