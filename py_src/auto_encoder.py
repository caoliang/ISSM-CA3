import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.random import seed
from tensorflow import set_random_seed

# Chips in 1 block which shares the same model
block_chips = 1

# training setup
EPOCHS = 40
BATCH_SIZE = 2

encoder_dimension = int(32 * 20 / block_chips)
layer1_dim = 640
layer2_dim = 320
layer3_dim = 10

def build_AE(x_train, encoder_dim=encoder_dimension):
    act_func = 'elu'

    # input placeholder
    signal_dim = x_train.shape[1]
    input_image = Input(shape=(signal_dim,))

    # encoding layer
    hidden_layer1 = Dense(layer1_dim, activation=act_func, kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(0.0))(input_image)

    hidden_layer2 = Dense(layer2_dim, activation=act_func, kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(0.0))(hidden_layer1)

    hidden_layer3 = Dense(layer3_dim, activation=act_func, kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(0.0))(hidden_layer2)

    hidden_layer4 = Dense(layer2_dim, activation=act_func, kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(0.0))(hidden_layer3)

    hidden_layer5 = Dense(layer1_dim, activation=act_func, kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(0.0))(hidden_layer4)

    # decoding layer
    decode_output = Dense(signal_dim, activation='relu')(hidden_layer3)

    # build autoencoder, encoder, decoder
    encoder = Model(inputs=input_image, outputs=hidden_layer2)
    autoencoder = Model(inputs=input_image, outputs=decode_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.05, verbose=0)

    return encoder, autoencoder


# Show the encoding coefficients

def show_encodings(inputs, outputs):
    n = len(inputs)
    # print('len(inputs): ', n)
    fig, axes = plt.subplots(2, n)
    for i in range(n):
        axes[0, i].imshow(inputs[i].reshape(-1, 20), cmap='gray')
        axes[1, i].imshow(outputs[i].reshape(-1, 20), cmap='gray')
    for ax in axes.flatten():
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# Perform prediction using the trained autoencoder model
def check_anomaly_and_compute_dist(ae_test_data, test_data_predict, threshold=10):
    print("In compute_dist, ae_test_data.shape:", ae_test_data.shape)

    sqrt_n = np.sqrt(ae_test_data.shape[1])

    dist = rmse_sqrt_n(ae_test_data, test_data_predict, sqrt_n)

    abnormal_status = dist > threshold

    abnormal_check_result = np.stack((abnormal_status, dist), axis=1)
    print("abnormal_check_result====", abnormal_check_result)
    '''
    if dist > threshold:
        print('Anomaly detected (distance: %.2f)' % dist)
        return dist, True
    else:
        return dist, False
    '''

def compute_dist(data_in, autoencoder_in, threshold=10):
    print(data_in.shape)

    data_dist = []
    sqrt_n = np.sqrt(data_in.shape[1])

    for i in range(len(data_in)):
        # Step 1: Select the test sample (such as the 0-th sample)
        test_data_orig = data_in[i:i + 1, :]

        # Step 2: Perform prediction using the trained autoencoder model
        test_data_predict = autoencoder_in.predict(test_data_orig)

        # Step 3: Calculate the reconstruction error and make decision on anomaly detection
        dist = rmse_sqrt_n(test_data_orig, test_data_predict, sqrt_n)
        data_dist.append(dist)

        if dist > threshold:
            print('Anomaly detected (distance: %.2f)' % dist)
        else:
            pass

    return np.array(data_dist)

def rmse_sqrt_n(y_data, y_data_pred, sqrt_n):
    return np.linalg.norm(y_data_pred - y_data, axis=-1) / sqrt_n