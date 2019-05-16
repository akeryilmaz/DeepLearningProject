import torch
import torchvision
import torch.nn as nn

class LoadedNetwork:
	def __init__(self, modelname, device):

		if modelname[:8] == "resnet18" or modelname[:8] == "Resnet18":
			self.net = torchvision.models.resnet18()
			self.net.fc = nn.Linear(self.net.fc.in_features, 47)
		elif modelname[:8] == "resnet50" or modelname[:8] == "Resnet50":
			self.net = torchvision.models.resnet50()
			self.net.fc = nn.Linear(self.net.fc.in_features, 47)
		elif modelname[:9] == "resnet101" or modelname[:9] == "Resnet101":
			self.net = torchvision.models.resnet101()
			self.net.fc = nn.Linear(self.net.fc.in_features, 47)

		elif modelname[:11] == "Densenet121" or modelname[:11] == "densenet121":
			self.net = torchvision.models.densenet121()
			self.net.classifier = nn.Linear(self.net.classifier.in_features, 47)
		elif modelname[:11] == "Densenet169" or modelname[:11] == "densenet169":
			self.net = torchvision.models.densenet169()
			self.net.classifier = nn.Linear(self.net.classifier.in_features, 47)
		elif modelname[:11] == "Densenet201" or modelname[:11] == "densenet201":
			self.net = torchvision.models.densenet201()
			self.net.classifier = nn.Linear(self.net.classifier.in_features, 47)

		elif modelname [:5] == "Vgg16" or modelname [:5] == "vgg16":
			self.net = torchvision.models.vgg16()
			self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, 47)
		elif modelname [:5] == "Vgg19" or modelname [:5] == "vgg19":
			self.net = torchvision.models.vgg19()
			self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, 47)

		else:
			raise Exception("Model not found.") 
		
		filename = "../Doc/Models/"+ modelname + ".pth"
		self.net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
		self.net.to(device)
		print('Model loaded from %s \n' % filename)

	def get_prediction(self, images):
		self.net.eval()
		return self.net(images)

