# This is project to classify car driving events into three categories: negligible, significant and critical.

### 1. *If you don`t have trained models* to make prediction you have to follow next points:
First of all download [driverBehaviorDataset](https://github.com/jair-jr/driverBehaviorDataset) and unarchive it.
<br>Clone current repository:
```
git clone https://gitlab.spd-ukraine.com/rnd-ml/car-driving-anomalies.git
```
Create ***data*** folder at the root of this repository and folder ***data_init*** inside it.
<br>After cloning ***copy content of the data folder from unarchived dataset to the data_init*** folder from the previous statement.
<br>Then you have to install all requirements. You can do this by the following command:
```
pip install -r requirements.txt
```
Run jupyter notebook:
```
jupyter notebook
```
####If you want to use accelerometer data only you have to run all cells next files:
***data_accelerometer.ipynb*** to transform data format to train the model.
<br>***filtering_accelerometer.ipynb*** to filter accelerometer data(with [Savitzky–Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)).
<br>***train_accelerometer_only.ipynb*** to train and save models.
####If you want to use accelerometer and gyroscope data you have to run all cells next files:
***data_accelerometer_gyroscope.ipynb*** to transform data format to train the model.
<br>***filtering_accelerometer_gyroscope.ipynb*** to filter accelerometer data(with [Savitzky–Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)).
<br>***train_accelerometer_gyroscope.ipynb*** to train and save models.

### 2. To make the prediction you have to run the following command:
```
python predict.py --model-file models/svc.pcl --file test.csv --saving-path .
```
#### Script has a few required parameters:
<li>model-file - path to the model (model must be saved in the pickle file).
<li>saving-path - path where csv-file with prediction should be saved.
<li>file - csv-file with data for predicting. Data should contain 6 columns: 

| column_name     | description                   | type  |
| ----------------|:-----------------------------:| -----:|
| x_accelerometer | accelerometer value by x-axis | float |
| y_accelerometer | accelerometer value by y-axis | float |
| z_accelerometer | accelerometer value by z-axis | float |
| x_gyroscope     | gyroscope value by x-axis     | float |
| y_gyroscope     | gyroscope value by y-axis     | float |
| z_gyroscope     | gyroscope value by z-axis     | float |
  
#### Parameters which you can change:

<li>model-type - type of the model. This argument is used for special preprocessing: for linear models accelerometer features will be normalized. By default the argument is linear. If you want to <b><i>use non-linear model</i></b> you have to <b><i>set model-type argument as non-linear</i></b>.
<li>window-length-accelerometer - parameter for filtering accelerometer values.
<li>polyorder-accelerometer - parameter for filtering accelerometer values.
<li>window-length-gyroscope - parameter for filtering gyroscope values.
<li>polyorder-gyroscope - parameter for filtering gyroscope values.
<li>gyroscope-feature - parameter to show using of gyroscope data as features.

##### Note that default filtering parameters are relevant if you use models which were trained as described in the [first statement](#1-if-you-dont-have-trained-models-to-make-prediction-you-have-to-follow-next-points). If you use your model trained with other filtering parameters you have to set the argument values when you run the script.