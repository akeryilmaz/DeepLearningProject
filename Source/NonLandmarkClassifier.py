import os
import numpy as np
import pandas as pd 
from PIL import Image
from cv2 import resize
import matplotlib.pyplot as plt
from vgg16_hybrid_places_1365 import *
#print(os.listdir("../Data/threshold"))

'''
# VGG 16 Places 365 scripts in custom dataset
os.chdir("/kaggle/input/keras-vgg16-places365/")
from vgg16_places_365 import VGG16_Places365
os.chdir("/kaggle/working/")
'''

# Get List of Images
data_dir = '../Data/threshold/'
class_information = pd.read_csv('../Doc/class_info_keras_classifier.csv')

'''
# Resize all images
all_images_resized = []
for filename in all_images:    
    im = np.array(Image.open(image_samples + filename).resize((224, 224), Image.LANCZOS))    
    all_images_resized.append(im)
'''
# Placeholders for predictions
p0, p1, p2 = [], [], []

# Places365 Model
model = VGG16_Hybrid_1365(weights='places')

# Loop through all images
for dirName, _, files in os.walk(data_dir):
	for filename in files:
		image_array = np.array(Image.open(dirName + "/" + filename))
		# Predict Top N Image Classes
		image_0 = np.expand_dims(image_array, 0)
		probs = np.sort(model.predict(image_0)[0])
		classes = np.argsort(model.predict(image_0)[0])
		classes = classes.tolist()
		probs = probs.tolist()
		cumProb = 0.0
		topClasses = []
		while cumProb < 0.95:
			topClasses.append(classes.pop())
			cumProb += probs.pop()


'''
  topn_preds = np.argsort(model.predict(image)[0])[::-1][0:topn]

  p0.append(topn_preds[0])
  p1.append(topn_preds[1])
  p2.append(topn_preds[2])

# Create dataframe for later usage
topn_df = pd.DataFrame()
topn_df['filename'] = np.array(all_images)
topn_df['p0'] = np.array(p0)
topn_df['p1'] = np.array(p1)
topn_df['p2'] = np.array(p2)
topn_df.to_csv('topn_class_numbers.csv', index = False)

# Summary
topn_df.head()


# Get 'landmark' images
n = 9
landmark_images =  topn_df[topn_df['p0_landmark'] == 'landmark']['filename']
landmark_indexes = landmark_images[:n].index.values

# Plot image examples
fig = plt.figure(figsize = (16, 16))
for index, im in zip(range(1, n+1), [ all_images_resized[i] for i in landmark_indexes]):
    fig.add_subplot(3, 3, index)
    plt.title(filename)
    plt.imshow(im)


# Get 'non-landmark' images
n = 9
landmark_images =  topn_df[topn_df['p0_landmark'] == 'non-landmark']['filename']
landmark_indexes = landmark_images[:n].index.values


# Plot image examples
fig = plt.figure(figsize = (16, 16))
for index, im in zip(range(1, n+1), [ all_images_resized[i] for i in landmark_indexes]):
    fig.add_subplot(3, 3, index)
    plt.title(filename)
    plt.imshow(im)

'''




