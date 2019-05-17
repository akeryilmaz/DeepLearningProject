import torch
import torchvision
import pandas as pd
import numpy as np

import torchvision.transforms as transforms

from NetworkLoader import *
from NonLandmarkClassifier import *

class NonlandmarkModel:
	def __init__(self, networkName, device):
		self.networkName = networkName
		self.net = LoadedNetwork(networkName, device)
		self.device = device
		self.trainHiddenStates = None

	def get_prediction(self, images):
		confidence, predicted = self.net.get_prediction(images)
		images_array = tensor_to_numpy(images)
		ifNonlandmarkList = check_nonlandmark(images_array)
		for i in range(images_array.shape[0]):
			if ifNonlandmarkList[i]:
				confidence[i] *= 0.1
		return confidence, predicted, ifNonlandmarkList
