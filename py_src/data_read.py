import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.random import seed
from tensorflow import set_random_seed

# # 1. Pre-processing chip temperature data

# Step 1. Read chip temperature data from CSV file
temp_base_data = pd.read_csv('./../data/temp_data.csv')
print("temp_base_data.shape:", temp_base_data.shape)

# Step 2. Divide data into time series data based on burn-in board coordinates
temp_data = temp_base_data.pivot(index='ID', columns='Time(min)')
print("temp_data.shape:", temp_data.shape)
# print(type(temp_data))

# Rename column name as string value
data_cols = list(temp_data.columns)
print("data_cols[0:1]:", data_cols[0:1])
cols_name_list = ['{0}_{1}'.format(col_item[0], col_item[1]) for col_item in data_cols]
print("cols_name_list[0:1]:", cols_name_list[0:1])
temp_data.columns = cols_name_list
print("len(list(temp_data.columns)):", len(list(temp_data.columns)))

# Step 3. Check Null value

null_value_sum = temp_data.isnull().sum().sum()
print('Total null value in dataset: ', null_value_sum)

# Step 4. Divide data to Time Series data and Chip ID data

temp_data_y = temp_data.index.values
temp_data_x = temp_data.values.reshape(400, 641, 20)

print('temp_data_x: ', temp_data_x.shape, 'temp_data_y: ', temp_data_y.shape)

# Step 5. Divide data into training set and testing set
x_train, x_test, Y_train, Y_test = train_test_split(temp_data_x, temp_data_y, test_size=0.20, random_state=49)
print("Training Data: ", x_train.shape, ", Training Index: ", Y_train.shape)
print("Testing Data: ", x_test.shape, ", Testing Index: ", Y_test.shape)