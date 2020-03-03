import numpy as np
import os
import pandas as pd
import pickle

from scipy.signal import savgol_filter


class Predictor(object):
    """
    Class for predicting.
    """

    def predict(self, data=None, linear=True, model_filename="svc.pcl", filtering=None, **kwargs):
        """
        :param data: data on which predict
        :param linear: linear model will be used or not (if True than do scaling if models are saved)
        :param model_filename: filename of the model (model have to be saved in the roor in "models" folder)
        :param filtering: filtering method
        :param kwargs: arguments for filtering
        :return: prediction on the data
        """
        if data is not None:
            with open(os.path.join("..", "models", model_filename), "rb") as file:
                model = pickle.load(file)
            if filtering is not None:
                data = self.preprocess(data, linear=linear, filtering=filtering, **kwargs)
            y_pred = model.predict(data).tolist()
            return y_pred

    @staticmethod
    def preprocess_feature(feature=None, scaler_filename=None, filtering=savgol_filter, **kwargs):
        """
        :param feature: feature vector which have to be preprocessed
        :param scaler_filename: filename of the stored scaler
        :param filtering: filtering method
        :param kwargs: arguments for filtering
        :return: preprocessed feature
        """
        if feature is not None:
            feature = filtering(feature, **kwargs)
            if scaler_filename is not None:
                with open(os.path.join("..", "models", scaler_filename), "rb") as file:
                    scaler = pickle.load(file)
                feature = scaler.transform(feature.reshape(-1, 1))
            return feature

    def preprocess(self, data=None, linear=False, filtering=savgol_filter, **kwargs):
        """
        :param data: data for preprocessing
        :param linear: linear model will be used or not (if True than do scaling if scalers are saved)
        :param filtering: filtering method
        :param kwargs: arguments for filtering
        :return:
        """
        if data is not None:
            data_new = pd.DataFrame()
            if linear:
                for column in data.columns:
                    # If linear models will be used look for scalers which were saved.
                    if os.path.exists(os.path.join("..", "models", f"{column}.pcl")):
                        scaler_filename = os.path.join("..", "models", f"{column}.pcl")
                    else:
                        scaler_filename = None
                    data_new[column] = self.preprocess_feature(feature=data[column], scaler_filename=scaler_filename,
                                                               filtering=filtering, **kwargs).reshape(-1, )
            else:
                for column in data.columns:
                    data_new[column] = self.preprocess_feature(feature=data[column], filtering=filtering, **kwargs)
            data_new['acceleration'] = np.sqrt(
                data_new['x_accelerometer'] ** 2 + data_new['y_accelerometer'] ** 2 + data_new['z_accelerometer'] ** 2)
            return data_new
