import os
import pandas as pd

from scipy.signal import savgol_filter
from predictor import Predictor

data = pd.read_csv(os.path.join('data', "train_accelerometer.csv"))
data = data.drop(["event"], axis=1)

predictor = Predictor()
"""
If you want use simple approach your data must contain one of the list of features written below:
1. [x_accelerometer, y_accelerometer, z_accelerometer]
2. [x_accelerometer, y_accelerometer, z_accelerometer, x_gyroscope, y_gyroscope, z_gyroscope]

If you want use article approach your data must contain one of the list of features written below:
1. [mean_window_x_accelerometer, mean_window_y_accelerometer, mean_window_z_accelerometer, 
    std_window_x_accelerometer, std_window_y_accelerometer, std_window_z_accelerometer,
    median_window_x_accelerometer, median_window_y_accelerometer, median_window_z_accelerometer,
    tendency_window_x_accelerometer, tendency_window_y_accelerometer, tendency_window_z_accelerometer]
2. [mean_window_x_accelerometer, mean_window_y_accelerometer, mean_window_z_accelerometer, 
    mean_window_x_gyroscope, mean_window_y_gyroscope, mean_window_z_gyroscope, 
    std_window_x_accelerometer, std_window_y_accelerometer, std_window_z_accelerometer,
    std_window_x_gyroscope, std_window_y_gyroscope, std_window_z_gyroscope,
    median_window_x_accelerometer, median_window_y_accelerometer, median_window_z_accelerometer,
    median_window_x_gyroscope, median_window_y_gyroscope, median_window_z_gyroscope,
    tendency_window_x_accelerometer, tendency_window_y_accelerometer, tendency_window_z_accelerometer,
    tendency_window_x_gyroscope, tendency_window_y_gyroscope, tendency_window_z_gyroscope]
More information about this feature you can read in article which was mentioned in README.

To control approach class have parameter "features": 
set it "simple" or "article" to choose simple or article approach respectively.

To control type of the model class have parameter "linear": 
set it True or False to choose linear or non-linear models will be used.
Note, that linear type of the model imply that you already have scalers 
which are saved in the folder "models" relative to the running file 
and with names respectively features it must scale.
(For example x_accelerometer.pcl will scale x_accelerometer feature)

To control file with model class have parameter "model_filename": 
you can write name of the file with model (it should be in the folder "models" relative to the running file)

To control filtering class have parameter "filtering": 
you can pass name of the imported function there.
Also you can pass all parameters for filtering function after parameter "filtering" of predictor class as written below.
Note, that article approach doesn't imply filtering, because it must be done during features creation.
"""
result = predictor.predict(data=data, linear=True, model_filename="linear-accelerometer.pcl", features="simple", filtering=savgol_filter, window_length=51, polyorder=5)
