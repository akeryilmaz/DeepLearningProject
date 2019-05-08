import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from shutil import move

top10_data = pd.read_csv('C:\\Users\\User\\Desktop\\Source\\top10.csv')

target_dir = "C:\\Users\\User\\Desktop\\top10"
root_dir = "C:\\Users\\User\\Desktop\\DLResized"
top10 = [138982, 62798, 177870, 176528, 192931, 126637, 83144, 171772, 20409, 151942]

i=0
c=0
for dirName, _, fileList in os.walk(root_dir):
    i +=1
    print (i)
    for file in fileList:
        imageID = file[:-4]
        landmark_id = top10_data[top10_data.id == imageID].landmark_id.get_values()
        if len(landmark_id) == 0:
            continue
        landmark_id = landmark_id[0]
        landmark_dir = target_dir + "\\" + str(landmark_id)
        if not os.path.exists(landmark_dir):
            print("created")
            os.mkdir(landmark_dir)
        c+=1
        print('counter: ' ,c )
        move(dirName + "\\" + file, landmark_dir + "\\" + imageID + ".jpg")
        '''landmark_id = train_data[train_data.id == imageID].landmark_id.get_values()[0]
        if landmark_id not in top10:
            continue
        c+=1
        print('counter: ' ,c )
        landmark_dir = target_dir + "\\" + str(landmark_id)
        if not os.path.exists(landmark_dir):
            os.mkdir(landmark_dir)
            move(dirName + "\\" + file, landmark_dir + "\\" + imageID + ".jpg")'''
