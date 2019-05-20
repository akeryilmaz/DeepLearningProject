import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from shutil import copy

target_dir = "../Data/splitData"
root_dir = "../Data/thresholdcopy"

pathList = []
for dirName, _, fileList in os.walk(root_dir):
	for file in fileList:
		pathList.append(dirName+ "/" + file)


random.shuffle(pathList)

lenList = len(pathList)

testList = pathList[:int(lenList/8)]
trainList = pathList[int(lenList/8):]

for test in testList:
	print("test  ",test)
	landmark_target = test.replace(root_dir, target_dir + "/test")
	landmark_id = landmark_target.split("/")[-2]
	class_path = target_dir + "/test/" + landmark_id
	if not os.path.exists(class_path):
		os.mkdir(class_path)
	copy(test, landmark_target)

for train in trainList:
	print("train  ",train)
	landmark_target = train.replace(root_dir, target_dir + "/train")
	landmark_id = landmark_target.split("/")[-2]
	class_path = target_dir + "/train/" + landmark_id
	if not os.path.exists(class_path):
		os.mkdir(class_path)
	copy(train, landmark_target)


