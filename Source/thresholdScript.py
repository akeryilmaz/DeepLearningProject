import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from shutil import move

NUM_THRESHOLD = 1000
train = pd.read_csv('C:\\Users\\User\\Desktop\\Source\\train.csv')

lands = pd.DataFrame(train.landmark_id.value_counts())
lands.reset_index(inplace=True)
lands.columns = ['landmark_id','count']

print (lands[lands['count'] >= NUM_THRESHOLD]['landmark_id'])
top_lands = set(lands[lands['count'] >= NUM_THRESHOLD]['landmark_id'])
print("Number of TOP classes {}".format(len(top_lands)))

temp = pd.DataFrame(columns = ["id", "url","landmark_id"])
for landmark in top_lands:
    temp = pd.concat([temp, train[train.landmark_id == landmark]])
temp.to_csv('C:\\Users\\User\\Desktop\\Source\\threshold.csv')