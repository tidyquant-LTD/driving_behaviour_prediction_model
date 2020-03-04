import numpy as np
import os
import pandas as pd
import pickle


class Predictor(object):
    """
    Class for predicting.
    """

    def predict(self, data=None, linear=True, model_filename="linear-accelerometer.pcl", features="simple", filtering=None, **kwargs):
        """
        :param data: data on which predict
        :param linear: linear model will be used or not (if True than do scaling if models are saved)
        :param model_filename: filename of the model (model have to be saved in the roor in "models" folder)
        :param filtering: filtering method
        :param kwargs: arguments for filtering
        :return: prediction on the data
        """
        if data is not None:
            with open(os.path.join("models", model_filename), "rb") as file:
                model = pickle.load(file)
            if filtering is not None:
                data = self.preprocess(data, linear=linear, filtering=filtering, **kwargs)
            if features == "simple":
                data['acceleration'] = np.sqrt(
                    data['x_accelerometer'] ** 2 + data['y_accelerometer'] ** 2 + data['z_accelerometer'] ** 2)
            y_pred = model.predict(data).tolist()
            return y_pred

    @staticmethod
    def preprocess_feature(feature=None, path_to_the_scaler=None, filtering=None, **kwargs):
        """
        :param feature: feature vector which have to be preprocessed
        :param scaler_filename: filename of the stored scaler
        :param filtering: filtering method
        :param kwargs: arguments for filtering
        :return: preprocessed feature
        """
        if feature is not None:
            feature = filtering(feature, **kwargs)
            if path_to_the_scaler is not None:
                with open(path_to_the_scaler, "rb") as file:
                    scaler = pickle.load(file)
                feature = scaler.transform(feature.reshape(-1, 1))
            return feature

    def preprocess(self, data=None, linear=False, filtering=None, **kwargs):
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
                    if os.path.exists(os.path.join("models", f"{column}.pcl")):
                        path_to_the_scaler = os.path.join("models", f"{column}.pcl")
                    else:
                        path_to_the_scaler = None
                    data_new[column] = self.preprocess_feature(feature=data[column], path_to_the_scaler=path_to_the_scaler,
                                                               filtering=filtering, **kwargs).reshape(-1, )
            else:
                for column in data.columns:
                    data_new[column] = self.preprocess_feature(feature=data[column], filtering=filtering, **kwargs)
            return data_new

    def predict_and_save(self, data=None, saving_path=None, linear=True,
                         model_filename="linear-accelerometer.pcl", filtering=None, **kwargs):
        if data is not None and saving_path is not None:
            data_new = data.copy()
            data_new['anomalies_category'] = self.predict(data=data, linear=linear, model_filename=model_filename,
                                                      filtering=filtering, **kwargs)
            data_new.to_csv(saving_path, index=False)
            print(f"File was saving by {saving_path}")
