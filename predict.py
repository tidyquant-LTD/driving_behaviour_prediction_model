import argparse
import pickle
import pandas as pd
import os

from numpy import sqrt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler


def preprocessing(args):
    data = pd.read_csv(args.file)

    # Filter accelerometer and gyroscope values
    data['x_accelerometer_fil'] = savgol_filter(data['x_accelerometer'].values, args.window_length_accelerometer,
                                                args.polyorder_accelerometer)
    data['y_accelerometer_fil'] = savgol_filter(data['y_accelerometer'].values, args.window_length_accelerometer,
                                                args.polyorder_accelerometer)
    data['z_accelerometer_fil'] = savgol_filter(data['z_accelerometer'].values, args.window_length_accelerometer,
                                                args.polyorder_accelerometer)

    data['acceleration_fil'] = sqrt(
        data['x_accelerometer_fil'] ** 2 + data['y_accelerometer_fil'] ** 2 + data['z_accelerometer_fil'] ** 2)

    if args.gyroscope_feature:
        data['x_gyroscope_fil'] = savgol_filter(data['x_gyroscope'].values, args.window_length_gyroscope,
                                                args.polyorder_gyroscope)
        data['y_gyroscope_fil'] = savgol_filter(data['y_gyroscope'].values, args.window_length_gyroscope,
                                                args.polyorder_gyroscope)
        data['z_gyroscope_fil'] = savgol_filter(data['z_gyroscope'].values, args.window_length_gyroscope,
                                                args.polyorder_gyroscope)

    # If model linear we must scale accelerometer data
    if args.model_type == "linear":
        with open('models/x_scaler.pcl', "rb") as file:
            x_scaler = pickle.load(file)

        with open('models/y_scaler.pcl', "rb") as file:
            y_scaler = pickle.load(file)

        with open('models/z_scaler.pcl', "rb") as file:
            z_scaler = pickle.load(file)

        data['x_accelerometer_fil_scaled'] = x_scaler.transform(data['x_accelerometer_fil'].values.reshape(-1, 1))
        data['y_accelerometer_fil_scaled'] = y_scaler.transform(data['y_accelerometer_fil'].values.reshape(-1, 1))
        data['z_accelerometer_fil_scaled'] = z_scaler.transform(data['z_accelerometer_fil'].values.reshape(-1, 1))
        data['acceleration_fil_scaled'] = sqrt(
            data['x_accelerometer_fil_scaled'] ** 2 + data['y_accelerometer_fil_scaled'] ** 2 + data[
                'z_accelerometer_fil_scaled'] ** 2)

    if args.gyroscope_feature:
        normalizer = MinMaxScaler()
        data['x_gyroscope_fil_scaled'] = normalizer.fit_transform(data['x_accelerometer_fil'].values.reshape(-1, 1))
        data['y_gyroscope_fil_scaled'] = normalizer.fit_transform(data['y_accelerometer_fil'].values.reshape(-1, 1))
        data['z_gyroscope_fil_scaled'] = normalizer.fit_transform(data['z_accelerometer_fil'].values.reshape(-1, 1))

    return data


def predict(args):
    data = preprocessing(args)

    with open(args.model_file, "rb") as file:
        model = pickle.load(file)

    with open("models/encoder.pcl", "rb") as file:
        encoder = pickle.load(file)

    # If model linear we must use scaled accelerometer data for predicting
    # Otherwise used filtered
    if args.model_type == "linear" and args.gyroscope_feature:
        data['anomalies_category'] = model.predict(data[["x_accelerometer_fil_scaled",
                                                         "y_accelerometer_fil_scaled",
                                                         "z_accelerometer_fil_scaled",
                                                         "acceleration_fil_scaled",
                                                         "x_gyroscope_fil_scaled",
                                                         "y_gyroscope_fil_scaled",
                                                         "z_gyroscope_fil_scaled"]].values)
    elif args.model_type == "linear":
        data['anomalies_category'] = model.predict(data[["x_accelerometer_fil_scaled",
                                                         "y_accelerometer_fil_scaled",
                                                         "z_accelerometer_fil_scaled",
                                                         "acceleration_fil_scaled"]].values)
    elif args.gyroscope_feature:
        data['anomalies_category'] = model.predict(data[["x_accelerometer_fil",
                                                         "y_accelerometer_fil",
                                                         "z_accelerometer_fil",
                                                         "acceleration_fil",
                                                         "x_gyroscope_fil",
                                                         "y_gyroscope_fil",
                                                         "z_gyroscope_fil"]].values)
    else:
        data['anomalies_category'] = model.predict(data[["x_accelerometer_fil",
                                                         "y_accelerometer_fil",
                                                         "z_accelerometer_fil",
                                                         "acceleration_fil"]].values)

    # Transform prediction to the readable form
    data['anomalies_category'] = encoder.inverse_transform(data['anomalies_category'].values)

    # Save prediction
    data.to_csv(os.path.join(args.saving_path, "car_driving_anomalies.csv"), index=False)
    print(f"File with predict was saving in {args.saving_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict anomalies in car driving.')
    parser.add_argument('--model-type', default='linear',
                        help='Type of the model for predicting. Should be "linear" or "non-linear".')
    parser.add_argument('--model-file', required=True,
                        help='Path to the file with model. Model have to be saved in the pickle file.')
    parser.add_argument('--file', required=True,
                        help='Path to the file with data.')
    parser.add_argument('--saving-path', required=True,
                        help='Path for saving file with predict.')
    parser.add_argument('--window-length-accelerometer', type=int, default=51,
                        help='Window length for filtering accelerometer values.')
    parser.add_argument('--polyorder-accelerometer', type=int, default=5,
                        help='Polyorder for filtering accelerometer values.')
    parser.add_argument('--window-length-gyroscope', type=int, default=31,
                        help='Window length for filtering gyroscope values.')
    parser.add_argument('--polyorder-gyroscope', type=int, default=4,
                        help='Polyorder for filtering gyroscope values.')
    parser.add_argument('--gyroscope-feature', type=bool, default=False,
                        help='Adding gyroscope values as features.')
    arguments = parser.parse_args()
    predict(arguments)
