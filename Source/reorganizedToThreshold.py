import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from shutil import move

thresholdClass_data = pd.read_csv('../Doc/thresholdClasses.csv')

target_dir = "../Data/threshold"
root_dir = "../Data/reorganized"

thresholdClasses = set(thresholdClass_data.landmark_id)
c=0

for dirName, _, fileList in os.walk(root_dir):
	current_id = dirName.split("/")[-1]
	try:
		int(current_id)
	except:
		continue 
	if int(current_id) in thresholdClasses:
		print(dirName)
		move(dirName, target_dir + "/" + current_id)
