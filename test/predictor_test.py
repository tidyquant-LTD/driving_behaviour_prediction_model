import numpy as np
import os
import pandas as pd
import pickle
import unittest

from predictor import Predictor
from scipy.signal import savgol_filter


class PredictorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = Predictor()

    def test_create_predictor(self):
        self.assertIsNotNone(self.predictor)

    def test_predict_method(self):
        self.predictor.predict()

    def test_predict_method_with_model_type(self):
        self.predictor.predict(linear=True)

    def test_predict_method_with_features_type(self):
        self.predictor.predict(features_type="simple")

    def test_predict_method_return_value(self):
        data = pd.read_csv(os.path.join("..", "data", "train_filtered_accelerometer.csv"))
        data['acceleration'] = np.sqrt(
            data['x_accelerometer'] ** 2 + data['y_accelerometer'] ** 2 + data['z_accelerometer'] ** 2)
        data = data.drop(['event'], axis=1)

        with open(os.path.join("..", "models", "non-linear-accelerometer.pcl"), "rb") as file:
            model = pickle.load(file)

        y_true = model.predict(data).tolist()[:10]
        y_pred = self.predictor.predict(data, linear=False, gyroscope=False,
                                        model_filename="non-linear-accelerometer.pcl", features="simple")[:10]

        self.assertListEqual(y_true, y_pred)

    def test_preprocess_feature_method(self):
        self.predictor.preprocess_feature()

    def test_preprocess_feature_method_with_filtering_and_parameters(self):
        self.predictor.preprocess_feature(filtering=savgol_filter, window_length=51, polyorder=5)

    def test_preprocess_feature_method_with_filtering_and_parameters_returned_value(self):
        window_length = 51
        polyorder = 5
        x = np.random.uniform(-1, 0, 100)
        filtered = savgol_filter(x, window_length=window_length, polyorder=polyorder).tolist()
        filtered_with_predictor = self.predictor.preprocess_feature(feature=x, filtering=savgol_filter,
                                                                    window_length=window_length,
                                                                    polyorder=polyorder).tolist()

        self.assertListEqual(filtered, filtered_with_predictor)

    def test_preprocess_feature_method_with_model_type(self):
        self.predictor.preprocess_feature(linear=True)

    def test_preprocess_feature_method_with_checking_model_type(self):
        with open(os.path.join("..", "models", "x_accelerometer.pcl"), "rb") as file:
            scaler = pickle.load(file)
        window_length = 51
        polyorder = 5
        x = np.random.uniform(-1, 0, 100)

        filtered = savgol_filter(x, window_length=window_length, polyorder=polyorder)
        scaled = scaler.transform(filtered.reshape(-1, 1)).tolist()

        scaled_with_predictor = self.predictor.preprocess_feature(feature=x,
                                                                  scaler_filename="x_accelerometer.pcl",
                                                                  filtering=savgol_filter,
                                                                  window_length=window_length,
                                                                  polyorder=polyorder).tolist()
        self.assertListEqual(scaled, scaled_with_predictor)

    def test_preprocess_method(self):
        self.predictor.preprocess()

    def test_preprocess_method_with_filtering(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer.csv"))
        data = data.drop(['event'], axis=1)

        window_length = 51
        polyorder = 5

        preprocessed_data = self.predictor.preprocess(data=data, filtering=savgol_filter, window_length=window_length,
                                                      polyorder=polyorder)
        result = preprocessed_data.loc[0].values.tolist()

        data_new = pd.DataFrame()
        for column in data.columns:
            data_new[column] = savgol_filter(data[column], window_length=window_length, polyorder=polyorder)

        true_values = data_new.loc[0].values.tolist()

        self.assertListEqual(true_values, result)

    def test_preprocess_method_with_filtering_and_scaling(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer.csv"))
        data = data.drop(['event'], axis=1)

        window_length = 51
        polyorder = 5

        preprocessed_data = self.predictor.preprocess(data=data, linear=True, filtering=savgol_filter,
                                                      window_length=window_length, polyorder=polyorder)
        result = preprocessed_data.loc[0].values.tolist()

        data_new = pd.DataFrame()
        for column in sorted(data.columns):
            data_new[column] = savgol_filter(data[column], window_length=window_length, polyorder=polyorder)
            with open(os.path.join("..", "models", f"{column}.pcl"), "rb") as file:
                scaler = pickle.load(file)
            data_new[column] = scaler.transform(data_new[column].values.reshape(-1, 1))

        true_values = data_new.loc[0].values.tolist()

        self.assertListEqual(true_values, result)

    def test_preprocess_method_with_gyroscope(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer_gyroscope.csv"))
        data = data.drop(['event'], axis=1)

        window_length = 51
        polyorder = 5

        preprocessed_data = self.predictor.preprocess(data=data, filtering=savgol_filter, window_length=window_length,
                                                      polyorder=polyorder)
        result = preprocessed_data.loc[0].values.tolist()

        data_new = pd.DataFrame()
        for column in data.columns:
            data_new[column] = savgol_filter(data[column], window_length=window_length, polyorder=polyorder)

        true_values = data_new.loc[0].values.tolist()

        self.assertListEqual(true_values, result)

    def test_predict_method_with_preprocess_filtering(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer.csv"))
        data = data.drop(['event'], axis=1)

        window_length = 51
        polyorder = 5

        with open(os.path.join("..", "models", "non-linear-accelerometer.pcl"), "rb") as file:
            model = pickle.load(file)
        data_new = pd.DataFrame()
        for column in data.columns:
            data_new[column] = savgol_filter(data[column], window_length=window_length, polyorder=polyorder)

        data_new['acceleration'] = np.sqrt(
            data_new['x_accelerometer'] ** 2 + data_new['y_accelerometer'] ** 2 + data_new['z_accelerometer'] ** 2)
        y_true = model.predict(data_new).tolist()[:10]
        y_pred = self.predictor.predict(data, linear=False, model_filename="non-linear-accelerometer.pcl",
                                        features="simple",
                                        filtering=savgol_filter,
                                        window_length=window_length, polyorder=polyorder)[:10]

        self.assertListEqual(y_true, y_pred)

    def test_predict_method_with_preprocess_filtering_and_scaling(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer.csv"))
        data = data.drop(['event'], axis=1)

        window_length = 51
        polyorder = 5

        with open(os.path.join("..", "models", "linear-accelerometer.pcl"), "rb") as file:
            model = pickle.load(file)

        data_new = pd.DataFrame()
        for column in data.columns:
            data_new[column] = savgol_filter(data[column], window_length=window_length, polyorder=polyorder)
            with open(os.path.join("..", "models", f"{column}.pcl"), "rb") as file:
                scaler = pickle.load(file)
            data_new[column] = scaler.transform(data_new[column].values.reshape(-1, 1))

        data_new['acceleration'] = np.sqrt(
            data_new['x_accelerometer'] ** 2 + data_new['y_accelerometer'] ** 2 + data_new['z_accelerometer'] ** 2)

        y_true = model.predict(data_new).tolist()[:50]
        y_pred = self.predictor.predict(data, linear=True, model_filename="linear-accelerometer.pcl",
                                        features="simple",
                                        filtering=savgol_filter,
                                        window_length=window_length, polyorder=polyorder)[:50]

        self.assertListEqual(y_true, y_pred)

    def test_predict_method_with_gyroscope(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer_gyroscope.csv"))
        data = data.drop(['event'], axis=1)

        window_length = 51
        polyorder = 5

        data_new = pd.DataFrame()
        for column in data.columns:
            data_new[column] = savgol_filter(data[column], window_length=window_length, polyorder=polyorder)
        data_new['acceleration'] = np.sqrt(
            data_new['x_accelerometer'] ** 2 + data_new['y_accelerometer'] ** 2 + data_new['z_accelerometer'] ** 2)

        with open(os.path.join("..", "models", "non-linear-accelerometer-gyroscope.pcl"), "rb") as file:
            model = pickle.load(file)

        y_true = model.predict(data_new).tolist()[:10]
        y_pred = self.predictor.predict(data, linear=False, model_filename="non-linear-accelerometer-gyroscope.pcl",
                                        features="simple",
                                        filtering=savgol_filter,
                                        window_length=window_length, polyorder=polyorder)[:10]

        self.assertListEqual(y_true, y_pred)

    def test_predict_method_with_gyroscope_preprocess_filtering_and_scaling(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer_gyroscope.csv"))
        data = data.drop(['event'], axis=1)

        window_length = 51
        polyorder = 5

        with open(os.path.join("..", "models", "linear-accelerometer-gyroscope.pcl"), "rb") as file:
            model = pickle.load(file)

        data_new = pd.DataFrame()
        for column in data.columns:
            data_new[column] = savgol_filter(data[column], window_length=window_length, polyorder=polyorder)
            if os.path.exists(os.path.join("..", "models", f"{column}.pcl")):
                with open(os.path.join("..", "models", f"{column}.pcl"), "rb") as file:
                    scaler = pickle.load(file)
                data_new[column] = scaler.transform(data_new[column].values.reshape(-1, 1))

        data_new['acceleration'] = np.sqrt(
            data_new['x_accelerometer'] ** 2 + data_new['y_accelerometer'] ** 2 + data_new['z_accelerometer'] ** 2)

        y_true = model.predict(data_new).tolist()[:50]
        y_pred = self.predictor.predict(data, linear=True, model_filename="linear-accelerometer-gyroscope.pcl",
                                        features="simple",
                                        filtering=savgol_filter,
                                        window_length=window_length, polyorder=polyorder)[:50]

        self.assertListEqual(y_true, y_pred)

    def test_predict_method_check_feature_type(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer_features.csv"))
        data = data.drop(['event'], axis=1)

        with open(os.path.join("..", "models", "non-linear-accelerometer-features.pcl"), "rb") as file:
            model = pickle.load(file)

        y_true = model.predict(data).tolist()[:10]
        y_pred = self.predictor.predict(data, model_filename="non-linear-accelerometer-features.pcl", features="article")[:10]

        self.assertListEqual(y_true, y_pred)

    def test_predict_and_save_method(self):
        self.predictor.predict_and_save()

    def test_predict_and_save_method_with_data(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer.csv"))
        data = data.drop(['event'], axis=1)

        self.predictor.predict_and_save(data=data)

    def test_predict_and_save_method_with_all_arguments_for_predict(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer.csv"))
        data = data.drop(['event'], axis=1)

        self.predictor.predict_and_save(data=data, linear=True, model_filename="linear-accelerometer.pcl",
                                        features="simple", filtering=savgol_filter, window_length=51, polyorder=5)

    def test_predict_and_save_method_with_all_arguments_for_predict_and_saving_path(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer.csv"))
        data = data.drop(['event'], axis=1)
        saving_path = "result.csv"

        self.predictor.predict_and_save(data=data, saving_path=saving_path, linear=True,
                                        model_filename="linear-accelerometer.pcl", features="simple",
                                        filtering=savgol_filter, window_length=51, polyorder=5)
        if os.path.exists(saving_path):
            os.remove(saving_path)

    def test_save_file_with_predict(self):
        data = pd.read_csv(os.path.join("..", "data", "train_accelerometer.csv"))
        data = data.drop(['event'], axis=1)

        with open(os.path.join("..", "models", "non-linear-accelerometer.pcl"), "rb") as file:
            model = pickle.load(file)

        data['acceleration'] = np.sqrt(
            data['x_accelerometer'] ** 2 + data['y_accelerometer'] ** 2 + data['z_accelerometer'] ** 2)
        y_true = model.predict(data).tolist()[:10]
        saving_filename = "test_file"
        saving_path = os.path.join("..", "data", f"{saving_filename}.csv")

        self.predictor.predict_and_save(data=data, saving_path=saving_path, linear=True,
                                        model_filename="non-linear-accelerometer.pcl",
                                        features="simple", filtering=None)

        data = pd.read_csv(saving_path)
        y_pred = data["anomalies_category"].tolist()[:10]

        self.assertListEqual(y_true, y_pred)
        if os.path.exists(saving_path):
            os.remove(saving_path)
