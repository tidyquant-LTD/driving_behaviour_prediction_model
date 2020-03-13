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
#### If you want to use accelerometer data only you have to run all cells in next files:
***data_accelerometer.ipynb*** to transform data format to train the model.
<br>***filtering_accelerometer.ipynb*** to filter accelerometer data(with [Savitzky–Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)).
<br>***train_accelerometer_only.ipynb*** to train and save models.
#### If you want to use accelerometer and gyroscope data you have to run all cells in next files:
***data_accelerometer_gyroscope.ipynb*** to transform data format to train the model.
<br>***filtering_accelerometer_gyroscope.ipynb*** to filter accelerometer data(with [Savitzky–Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)).
<br>***train_accelerometer_gyroscope.ipynb*** to train and save models.
#### If you want to use accelerometer data with approach described in the [article](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0174959) you have to run all cells in next files:
***data_accelerometer_features.ipynb*** to transform data format to train the model.
<br>***train_accelerometer_only_features.ipynb*** to train and save models.
#### If you want to use accelerometer and gyroscope data with approach described in the [article](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0174959) you have to run all cells in next files:
***data_accelerometer_gyroscope_features.ipynb*** to transform data format to train the model.
<br>***train_accelerometer_gyroscope_features.ipynb*** to train and save models.
##### Models will be saved after training(you can change the models you want to save or their names in the last cells for prediction files). Scalers and encoder will also be saved, these attributes will be used for prediction in future.
### 2. Example for making a prediction is in the ***example.py*** file. Also this file contains description of Predictor class parameters and features needed to be used by a certain model.
Note that each model works with its own set of features and if the file which you are loading contains a less or a bigger number of features, this will cause an error.
### 3. You can run tests for Predictor class with the following command:
```
python -m unittest tests.py
```
Tests can be run only after running all cells in all files mentioned in [Statement #1](#1-if-you-dont-have-trained-models-to-make-prediction-you-have-to-follow-next-points), because there will be a usability test of models which work with any set of features.
### 4. The repository has files which we used for visualization and analysis of the results, to work with the test data.
***confidence_matrix.ipynb*** to analyzing models result with confidence matrix, recall and precision scores.
<br>***predict_visualization.ipynb*** to visualizate predicted result on map.
<br>***test_predict.py*** to predict on test data with all models an save results.
<br>***test_data_transform.py*** to transform test data format to train the model.