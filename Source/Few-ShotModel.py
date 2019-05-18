import torch
import torchvision
import pandas as pd
import numpy as np

import torchvision.transforms as transforms

from NetworkLoader import *

class Few_Shot_Model:
	def __init__(self, networkName, device):
		self.networkName = networkName
		self.net = LoadedNetwork(networkName, device)
		self.device = device
		self.trainHiddenStates = None

	def load_train_hidden_states(self, trainDataDir = "../Data/splitDataNonlandmark/train"):
		try:
			trainHiddenStates = pd.read_csv("../Doc/Hiddens/" + self.networkName + ".csv")
			print("Train Data Hidden States are loaded.")
		except:
			normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
																	std=[0.229, 0.224, 0.225])
			train_transforms = transforms.Compose([transforms.Resize(224),
																		transforms.ToTensor(),
																		normalize,
																		])
			train_data = torchvision.datasets.ImageFolder(trainDataDir,
										transform=train_transforms)
			trainloader = torch.utils.data.DataLoader(train_data,
										shuffle=False, batch_size=512)
			trainHiddenStates = pd.DataFrame(columns = ["hiddenState", "label"])
			with torch.no_grad():
				for i, (images, labels) in enumerate(trainloader):
					images, labels = images.to(self.device), labels.to(self.device)
					outputs = self.net.get_hidden_state(images)
					outputs = [tuple(row) for row in outputs.cpu().numpy()]
					temp = pd.DataFrame({"hiddenState": outputs, "label":labels.cpu().numpy()})
					trainHiddenStates = trainHiddenStates.append(temp, ignore_index = True)	
			trainHiddenStates.to_csv("../Doc/Hiddens/" + self.networkName + ".csv")
			print("Train Data Hidden States are written to file.")
		self.trainHiddenStates = trainHiddenStates

	def find_k_closest(self):
		if self.trainHiddenStates == None:
			raise Exception("First load train data hidden states.")
		else:
			pass

	def get_prediction(self, images):
		if self.trainHiddenStates == None:
			raise Exception("First load train data hidden states.")
		else:
			pass
		hiddenStates = self.net.get_hidden_state(images)

if __name__ == "__main__":
	model = Few_Shot_Model("densenet121SplitDataWithNonlandmark", torch.device('cuda:0'))
	model.load_train_hidden_states()
