import torch
import torchvision
import torch.nn as nn

class Network:
	def __init__(self, net, device, modeltype=None):
		self.net = net
		self.modeltype = modeltype
		self.net.to(device)

	def get_prediction(self, images):
		self.net.eval()
		outputs = self.net(images)
		min, _ = torch.min(outputs.data, 1)
		outputs -= min.reshape(images.shape[0], 1).expand(-1, outputs.data.shape[1])
		max, predicted = torch.max(outputs.data, 1)
		confidence = max/torch.sum(outputs.data, 1)
		return confidence, predicted

	def set_modeltype(self, modeltype):
		self.modeltype = modeltype

	def get_hidden_state(self, images):
		self.net.eval()
		feature_extractor = self.net
		if self.modeltype == "resnet": 
			feature_extractor.fc = torch.nn.Sequential()
		elif self.modeltype == "densenet": 
			feature_extractor.classifier = torch.nn.Sequential()
		elif self.modeltype == "vgg": 
			feature_extractor.classifier[6] = torch.nn.Sequential()
		else:
			raise Exception("ModeltypeNotFound")
		hiddenStates = feature_extractor(images)
		return hiddenStates
