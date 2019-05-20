import os
import numpy as np
import pandas as pd 
from PIL import Image
from cv2 import resize
import matplotlib.pyplot as plt
from shutil import move
from vgg16_places_365 import *
import torchvision.transforms as transforms

def check_nonlandmark(images):
	model = VGG16_Places365(weights='places')
	data = pd.read_csv('../Doc/iflandmark.csv')

	probs = np.sort(predictions)
	classes = np.argsort(predictions)

	result = []
	for i in range(images.shape[0]):
		classesForImage = classes[i].tolist()
		probsForImage = probs[i].tolist()

		nonlandmarkProb = 0.0
		landmarkProb = 0.0
		ifNonLandmark = False
		while landmarkProb < 0.15 :
			if classesForImage == []:
				break
			currentClass = classesForImage.pop()
			iflandmark = data[data.index == currentClass].iflandmark.get_values()[0]
			if iflandmark == 1:
				nonlandmarkProb += probsForImage.pop()
			else:
				landmarkProb += probsForImage.pop()
			if nonlandmarkProb >= 0.85:
				      ifNonLandmark = True
		result.append(ifNonLandmark)
	return result

def tensor_to_numpy(images):
	inverse = transforms.Compose([transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0/0.229, 1.0/0.224, 1.0/0.225]), 
																transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0])])
	image_array = [inverse(image.cpu()).permute(1,2,0).numpy() for image in images]
	return np.array(image_array)
	

if __name__ == "__main__":
	# Get List of Images
	data_dir = '../Data/splitData/train'
	nonlandmark_dir = '../Data/splitData/nonlandmarkTrain'
	# Loop through all images
	for dirName, _, files in os.walk(data_dir):
		print (dirName)
		for filename in files:
			image_array = np.array(Image.open(dirName + "/" + filename))
			image_0 = np.expand_dims(image_array, 0)
			if(check_nonlandmark(image_0)):
					landmark_id = dirName.split("/")[-1]
					class_path = nonlandmark_dir + "/" + landmark_id
					if not os.path.exists(class_path):
						os.mkdir(class_path)
					move(dirName + "/" + filename,  class_path + "/" + filename)			
					break
		






