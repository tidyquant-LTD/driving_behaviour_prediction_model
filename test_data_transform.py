import os
import pandas as pd
from copy import copy

listdir = os.listdir(os.path.join('data', 'test'))

if not os.path.exists(os.path.join('data', 'test', 'transformed')):
    os.makedirs(os.path.join('data', 'test', 'transformed'))

for folder in listdir:

    data_acc = pd.read_csv(os.path.join('data', 'test',  folder, 'accelerometer.csv'))
    data_gyr = pd.read_csv(os.path.join('data', 'test',  folder, 'gyroscope.csv'))
    data_acc = data_acc.rename(columns={"x": "x_accelerometer", "y": "y_accelerometer", "z": "z_accelerometer"})
    y_temp = copy(data_acc['y_accelerometer'].values)
    z_temp = copy(data_acc['z_accelerometer'].values)
    data_acc['y_accelerometer'] = z_temp
    data_acc['z_accelerometer'] = y_temp
    data_acc['z_accelerometer'] = data_acc['z_accelerometer'] - 9.8
    data_gyr = data_gyr.rename(columns={"x": "x_gyroscope", "y": "y_gyroscope", "z": "z_gyroscope"})
    data_acc_gyr = pd.concat([data_acc, data_gyr], axis=1)
    y_temp = copy(data_acc_gyr['y_gyroscope'].values)
    z_temp = copy(data_acc_gyr['z_gyroscope'].values)
    data_acc_gyr['y_gyroscope'] = z_temp
    data_acc_gyr['z_gyroscope'] = y_temp
    data_acc.to_csv(os.path.join('data', 'test', 'transformed', f"{folder}_accelerometer.csv"), index=False)
    data_acc_gyr.to_csv(os.path.join('data', 'test', 'transformed', f"{folder}_accelerometer_gyroscope.csv"), index=False)
