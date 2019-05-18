import torch
import torchvision
import torch.nn as nn
from Network import *

class LoadedNetwork:
	def __init__(self, modelname, device):
		
		if modelname[:8] == "resnet18" or modelname[:8] == "Resnet18":
			self.net = torchvision.models.resnet18()
			self.net.fc = nn.Linear(self.net.fc.in_features, 47)
			self.modeltype = "resnet" 
		elif modelname[:8] == "resnet50" or modelname[:8] == "Resnet50":
			self.net = torchvision.models.resnet50()
			self.net.fc = nn.Linear(self.net.fc.in_features, 47)
			self.modeltype = "resnet" 
		elif modelname[:9] == "resnet101" or modelname[:9] == "Resnet101":
			self.net = torchvision.models.resnet101()
			self.net.fc = nn.Linear(self.net.fc.in_features, 47)
			self.modeltype = "resnet" 

		elif modelname[:11] == "Densenet121" or modelname[:11] == "densenet121":
			self.net = torchvision.models.densenet121()
			self.net.classifier = nn.Linear(self.net.classifier.in_features, 47)
			self.modeltype = "densenet" 
		elif modelname[:11] == "Densenet169" or modelname[:11] == "densenet169":
			self.net = torchvision.models.densenet169()
			self.net.classifier = nn.Linear(self.net.classifier.in_features, 47)
			self.modeltype = "densenet" 
		elif modelname[:11] == "Densenet201" or modelname[:11] == "densenet201":
			self.net = torchvision.models.densenet201()
			self.net.classifier = nn.Linear(self.net.classifier.in_features, 47)
			self.modeltype = "densenet" 

		elif modelname [:5] == "Vgg16" or modelname [:5] == "vgg16":
			self.net = torchvision.models.vgg16()
			self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, 47)
			self.modeltype = "vgg" 
		elif modelname [:5] == "Vgg19" or modelname [:5] == "vgg19":
			self.net = torchvision.models.vgg19()
			self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, 47)
			self.modeltype = "vgg" 

		else:
			raise Exception("Model not found.") 
		
		filename = "../Doc/Models/"+ modelname + ".pth"
		self.net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
		self.net.to(device)
		print('Model loaded from %s \n' % filename)

	def get_prediction(self, images):
		self.net.eval()
		outputs = self.net(images)
		min, _ = torch.min(outputs.data, 1)
		outputs -= min.reshape(images.shape[0], 1).expand(-1, outputs.data.shape[1])
		max, predicted = torch.max(outputs.data, 1)
		confidence = max/torch.sum(outputs.data, 1)
		return confidence, predicted

	def get_hidden_state(self, images):
		self.net.eval()
		feature_extractor = self.net
		if self.modeltype == "resnet": 
			feature_extractor.fc = torch.nn.Sequential()
		if self.modeltype == "densenet": 
			feature_extractor.classifier = torch.nn.Sequential()
		if self.modeltype == "vgg": 
			feature_extractor.classifier[6] = torch.nn.Sequential()
		hiddenStates = feature_extractor(images)
		return hiddenStates
