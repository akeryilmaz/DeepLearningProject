import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from shutil import move
import seaborn as sns

train_data = pd.read_csv('../input/train.csv')

target_dir = "../Data/reorganized"
root_dir = "../Data/resized"

for dirName, _, fileList in os.walk(root_dir):
	for file in fileList:
		imageID = file[:-4]
		train_data[train_data.id == imageID]
		landmark_id = str(train_data[train_data.id == imageID].landmark_id.get_values()[0])
		landmark_dir = target_dir + "/" + landmark_id 
		if not os.path.exists(landmark_dir):
			os.mkdir(landmark_dir)
		move(dirName + "/" + file, landmark_dir + "/" + imageID + ".jpg")


