import os
import numpy as np
import pandas as pd 
from PIL import Image
from cv2 import resize
import matplotlib.pyplot as plt
from shutil import move
from vgg16_places_365 import *
#print(os.listdir("../Data/threshold"))

# Get List of Images
data_dir = '../Data/splitData/train'
class_information = pd.read_csv('../Doc/class_info_keras_classifier.csv')
nonlandmark_dir = '../Data/splitData/nonlandmarkTrain'
# Places365 Model
model = VGG16_Places365(weights='places')
data = pd.read_csv('../Doc/iflandmark.csv')

# Loop through all images
for dirName, _, files in os.walk(data_dir):
	print (dirName)
	for filename in files:
		image_array = np.array(Image.open(dirName + "/" + filename))
		# Predict Top N Image Classes
		image_0 = np.expand_dims(image_array, 0)
		probs = np.sort(model.predict(image_0)[0])
		classes = np.argsort(model.predict(image_0)[0])
		classes = classes.tolist()
		probs = probs.tolist()
		nonlandmarkProb = 0.0
		landmarkProb = 0.0
		while landmarkProb < 0.15 :
			currentClass = classes.pop()
			iflandmark = data[data.index == currentClass].iflandmark.get_values()[0]
			if iflandmark == 1:
				nonlandmarkProb += probs.pop()
			else:
				landmarkProb += probs.pop()
			if nonlandmarkProb >= 0.85:
				#TODO copy the file to somewhere else
				landmark_id = dirName.split("/")[-1]
				class_path = nonlandmark_dir + "/" + landmark_id
				if not os.path.exists(class_path):
					os.mkdir(class_path)
				move(dirName + "/" + filename,  class_path + "/" + filename)			
				break
		#print(cumProb, "\n")
		#print(topClasses, "\n")
		





