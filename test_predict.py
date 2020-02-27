import os
import subprocess

listdir = os.listdir(os.path.join('data', 'test', 'transformed'))

if not os.path.exists(os.path.join('data', 'test', 'predict')):
    os.makedirs(os.path.join('data', 'test', 'predict'))
    os.makedirs(os.path.join('data', 'test', 'predict', 'linear'))
    os.makedirs(os.path.join('data', 'test', 'predict', 'non-linear'))

accelerometer_files = [file for file in listdir if "gyroscope" not in file]
accelerometer_gyroscope_files = [file for file in listdir if "gyroscope" in file]

for file in accelerometer_files:
    subprocess.call(f"venv/bin/python predict.py --model-file models/svc.pcl --file {os.path.join('data', 'test', 'transformed', file)} --saving-path {os.path.join('data', 'test', 'predict', 'linear')} --output-filename {file[:-4]}", shell=True)
    subprocess.call(f"venv/bin/python predict.py --model-file models/random_forest.pcl --model-type non-linear --file {os.path.join('data', 'test', 'transformed', file)} --saving-path {os.path.join('data', 'test', 'predict', 'non-linear')} --output-filename {file[:-4]}", shell=True)

for file in accelerometer_gyroscope_files:
    subprocess.call(f"venv/bin/python predict.py --model-file models/svc_gyroscope.pcl --file {os.path.join('data', 'test', 'transformed', file)} --saving-path {os.path.join('data', 'test', 'predict', 'linear')} --output-filename {file[:-4]} --gyroscope-feature True", shell=True)
    subprocess.call(f"venv/bin/python predict.py --model-file models/random_forest_gyroscope.pcl --model-type non-linear --file {os.path.join('data', 'test', 'transformed', file)} --saving-path {os.path.join('data', 'test', 'predict', 'non-linear')} --output-filename {file[:-4]} --gyroscope-feature True", shell=True)