import os
import pandas as pd

from scipy.signal import butter, lfilter
from predictor import Predictor


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


listdir = os.listdir(os.path.join('data', 'test', 'transformed'))

if not os.path.exists(os.path.join('data', 'test', 'predict')):
    os.makedirs(os.path.join('data', 'test', 'predict'))
    os.makedirs(os.path.join('data', 'test', 'predict', 'linear'))
    os.makedirs(os.path.join('data', 'test', 'predict', 'non-linear'))
    os.makedirs(os.path.join('data', 'test', 'predict', 'linear-features'))
    os.makedirs(os.path.join('data', 'test', 'predict', 'non-linear-features'))

accelerometer_files = [file for file in listdir if "gyroscope" not in file and "features" not in file]
accelerometer_gyroscope_files = [file for file in listdir if "gyroscope" in file and "features" not in file]
accelerometer_feature_files = [file for file in listdir if "gyroscope" not in file and "features" in file]
accelerometer_gyroscope_features_files = [file for file in listdir if "gyroscope" in file and "features" in file]

predictor = Predictor()
for file in accelerometer_files:
    data = pd.read_csv(os.path.join('data', 'test', 'transformed', file))

    saving_path = os.path.join('data', 'test', 'predict', 'linear', file)
    predictor.predict_and_save(data=data, saving_path=saving_path, linear=True, model_filename="linear-accelerometer.pcl", features="simple", filtering=butter_lowpass_filter, cutoff=2, fs=1000)

    saving_path = os.path.join('data', 'test', 'predict', 'non-linear', file)
    predictor.predict_and_save(data=data, saving_path=saving_path, linear=False, model_filename="non-linear-accelerometer.pcl", features="simple", filtering=butter_lowpass_filter, cutoff=2, fs=1000)

for file in accelerometer_gyroscope_files:
    data = pd.read_csv(os.path.join('data', 'test', 'transformed', file))

    saving_path = os.path.join('data', 'test', 'predict', 'linear', file)
    predictor.predict_and_save(data=data, saving_path=saving_path, linear=True, model_filename="linear-accelerometer-gyroscope.pcl", features="simple", filtering=butter_lowpass_filter, cutoff=2, fs=1000)

    saving_path = os.path.join('data', 'test', 'predict', 'non-linear', file)
    predictor.predict_and_save(data=data, saving_path=saving_path, linear=False, model_filename="non-linear-accelerometer-gyroscope.pcl", features="simple", filtering=butter_lowpass_filter, cutoff=2, fs=1000)

for file in accelerometer_feature_files:
    data = pd.read_csv(os.path.join('data', 'test', 'transformed', file))

    saving_path = os.path.join('data', 'test', 'predict', 'linear-features', file)
    predictor.predict_and_save(data=data, saving_path=saving_path, model_filename="linear-accelerometer-features.pcl", features="article")

    saving_path = os.path.join('data', 'test', 'predict', 'non-linear-features', file)
    predictor.predict_and_save(data=data, saving_path=saving_path, model_filename="non-linear-accelerometer-features.pcl", features="article")

for file in accelerometer_gyroscope_features_files:
    data = pd.read_csv(os.path.join('data', 'test', 'transformed', file))

    saving_path = os.path.join('data', 'test', 'predict', 'linear-features', file)
    predictor.predict_and_save(data=data, saving_path=saving_path, model_filename="linear-accelerometer-gyroscope-features.pcl", features="article")

    saving_path = os.path.join('data', 'test', 'predict', 'non-linear-features', file)
    predictor.predict_and_save(data=data, saving_path=saving_path, model_filename="non-linear-accelerometer-gyroscope-features.pcl", features="article")