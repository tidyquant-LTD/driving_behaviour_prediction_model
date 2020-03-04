import numpy as np
import os
import pandas as pd

from copy import copy
from scipy.signal import butter, lfilter


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def roll_column_with_duplicate(column):
    result = np.roll(column, 1)
    result[0] = result[1]
    return result

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

    data_acc = data_acc.drop(["time"], axis=1)
    data_acc_gyr = data_acc_gyr.drop(["time"], axis=1)

    data_acc.to_csv(os.path.join('data', 'test', 'transformed', f"{folder}_accelerometer.csv"), index=False)
    data_acc_gyr.to_csv(os.path.join('data', 'test', 'transformed', f"{folder}_accelerometer_gyroscope.csv"), index=False)

    data_acc['x_accelerometer'] = butter_lowpass_filter(data_acc['x_accelerometer'].values, 2, 1000)
    data_acc['y_accelerometer'] = butter_lowpass_filter(data_acc['y_accelerometer'].values, 2, 1000)
    data_acc['z_accelerometer'] = butter_lowpass_filter(data_acc['z_accelerometer'].values, 2, 1000)
    data_gyr["x_gyroscope"] = butter_lowpass_filter(data_gyr['x_gyroscope'].values, 2, 1000)
    data_gyr["y_gyroscope"] = butter_lowpass_filter(data_gyr['y_gyroscope'].values, 2, 1000)
    data_gyr["z_gyroscope"] = butter_lowpass_filter(data_gyr['z_gyroscope'].values, 2, 1000)

    data_acc["mean_window_x_accelerometer"] = data_acc["x_accelerometer"].rolling(8, min_periods=1).mean()
    data_acc["mean_window_y_accelerometer"] = data_acc["y_accelerometer"].rolling(8, min_periods=1).mean()
    data_acc["mean_window_z_accelerometer"] = data_acc["z_accelerometer"].rolling(8, min_periods=1).mean()
    data_gyr["mean_window_x_gyroscope"] = data_gyr["x_gyroscope"].rolling(8, min_periods=1).mean()
    data_gyr["mean_window_y_gyroscope"] = data_gyr["y_gyroscope"].rolling(8, min_periods=1).mean()
    data_gyr["mean_window_z_gyroscope"] = data_gyr["z_gyroscope"].rolling(8, min_periods=1).mean()

    data_acc["std_window_x_accelerometer"] = data_acc["x_accelerometer"].rolling(8, min_periods=1).std()
    data_acc["std_window_y_accelerometer"] = data_acc["y_accelerometer"].rolling(8, min_periods=1).std()
    data_acc["std_window_z_accelerometer"] = data_acc["z_accelerometer"].rolling(8, min_periods=1).std()
    data_gyr["std_window_x_gyroscope"] = data_gyr["x_gyroscope"].rolling(8, min_periods=1).std()
    data_gyr["std_window_y_gyroscope"] = data_gyr["y_gyroscope"].rolling(8, min_periods=1).std()
    data_gyr["std_window_z_gyroscope"] = data_gyr["z_gyroscope"].rolling(8, min_periods=1).std()

    data_acc["median_window_x_accelerometer"] = data_acc["x_accelerometer"].rolling(8, min_periods=1).median()
    data_acc["median_window_y_accelerometer"] = data_acc["y_accelerometer"].rolling(8, min_periods=1).median()
    data_acc["median_window_z_accelerometer"] = data_acc["z_accelerometer"].rolling(8, min_periods=1).median()
    data_gyr["median_window_x_gyroscope"] = data_gyr["x_gyroscope"].rolling(8, min_periods=1).median()
    data_gyr["median_window_y_gyroscope"] = data_gyr["y_gyroscope"].rolling(8, min_periods=1).median()
    data_gyr["median_window_z_gyroscope"] = data_gyr["z_gyroscope"].rolling(8, min_periods=1).median()

    data_acc["tendency_window_x_accelerometer"] = roll_column_with_duplicate(data_acc["mean_window_x_accelerometer"].values) / data_acc["mean_window_x_accelerometer"]
    data_acc["tendency_window_y_accelerometer"] = roll_column_with_duplicate(data_acc["mean_window_y_accelerometer"].values) / data_acc["mean_window_y_accelerometer"]
    data_acc["tendency_window_z_accelerometer"] = roll_column_with_duplicate(data_acc["mean_window_z_accelerometer"].values) / data_acc["mean_window_z_accelerometer"]
    data_gyr["tendency_window_x_gyroscope"] = roll_column_with_duplicate(data_gyr["mean_window_x_gyroscope"].values) / data_gyr["mean_window_x_gyroscope"]
    data_gyr["tendency_window_y_gyroscope"] = roll_column_with_duplicate(data_gyr["mean_window_y_gyroscope"].values) / data_gyr["mean_window_y_gyroscope"]
    data_gyr["tendency_window_z_gyroscope"] = roll_column_with_duplicate(data_gyr["mean_window_z_gyroscope"].values) / data_gyr["mean_window_z_gyroscope"]

    data_acc = data_acc.fillna(method="bfill")
    data_gyr = data_gyr.fillna(method="bfill")

    data_acc = data_acc.drop(["x_accelerometer", "y_accelerometer", "z_accelerometer"], axis=1)
    data_acc.to_csv(os.path.join('data', 'test', 'transformed', f"{folder}_accelerometer_features.csv"), index=False)
    data = pd.concat([data_acc, data_gyr], axis=1)
    data = data.drop(["time", 'x_gyroscope', 'y_gyroscope', 'z_gyroscope'], axis=1)
    data.to_csv(os.path.join('data', 'test', 'transformed', f"{folder}_accelerometer_gyroscope_features.csv"), index=False)
