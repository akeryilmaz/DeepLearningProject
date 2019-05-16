import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

BATCHSIZE = 256

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
																		std=[0.229, 0.224, 0.225])
TESTTRANSFORMS = transforms.Compose([transforms.Resize(224),
																			transforms.ToTensor(),
																			normalize,
																			])

RESNETS = ["resnet18-1", "resnet18SplitData","resnet18Epoch30", "resnet18SplitWithNonlandmarkData", "resnet18SplitWithNonlandmarkDataBatchsize256", "resnet50Epoch10", "resnet50Milestone", "resnet101Milestone"]

DENSENETS = ["Densenet121Milestone"]

VGGS = ["Vgg16Milestone"]


device = torch.device('cuda:0')
#device = torch.device('cpu')

if os.path.isdir('../Data/threshold'):
        data_dir = '../Data/threshold'
else:
        #TODO Change the error given here.
        raise Exception("Data directry not found!")

def load_test(datadir):
        test_transforms = TESTTRANSFORMS
        test_data = torchvision.datasets.ImageFolder(datadir,
								        transform=test_transforms)
        testloader = torch.utils.data.DataLoader(test_data,
								        shuffle=True, batch_size=BATCHSIZE)
        return testloader

def compute_accuracy_and_GAP(net, testloader, return_result):
	net.eval()
	correct = 0
	total = 0
	result = pd.DataFrame(columns = ["pred", "conf", "true"])
	with torch.no_grad():
		for i, (images, labels) in enumerate(testloader):
			images, labels = images.to(device), labels.to(device)
			outputs = net(images)
			confidence, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			temp = pd.DataFrame({"pred":predicted.cpu(), "conf":confidence.cpu(), "true":labels.cpu()})
			result = result.append(temp, ignore_index = True)
	result.sort_values('conf', ascending=False, inplace=True, na_position='last')
	result['correct'] = (result.true == result.pred).astype(int)
	result['prec_k'] = result.correct.cumsum() / (np.arange(len(result)) + 1)
	result['term'] = result.prec_k * result.correct
	gap = result.term.sum() / result.true.count()
	if return_result:
		return correct / total, gap, result
	else:
		return correct / total, gap




testloader = load_test(data_dir)

for RESNET in RESNETS:
	filename = "../Doc/Models/"+ RESNET + ".pth"
	if RESNET[:8] == "resnet18" or RESNET[:8] == "Resnet18":
		MODEL = torchvision.models.resnet18
	else if RESNET[:8] == "resnet50" or RESNET[:8] == "Resnet50":
		MODEL = torchvision.models.resnet50
	else:
		MODEL = torchvision.models.resnet101

	net = MODEL()
	net.fc = nn.Linear(net.fc.in_features, 47)
	net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
	net.to(device)
	print('Model loaded from %s \n' % filename)

	accuracy, gap, result = compute_accuracy_and_GAP(net, testloader, return_result = True)
	print('Accuracy of the network on the test images: %.3f' % accuracy)
	print('GAP of the network on the test images: %.3f' % gap)
	result.to_csv("../Doc/Models/" + RESNET + ".csv" )
        
for DENSENET in DENSENETS:
	if DENSENET[:11] == "Densenet121" or DENSENET[:11] == "densenet121":
		MODEL = torchvision.models.densenet121
	if DENSENET[:11] == "Densenet169" or DENSENET[:11] == "densenet169":
		MODEL = torchvision.models.densenet169
	else:
		MODEL = torchvision.models.densenet201
	filename = "../Doc/Models/"+ DENSENET + ".pth"

	net = MODEL(pretrained = PRETRAINED)
	net.classifier = nn.Linear(net.classifier.in_features, 47)
	net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
	net.to(device)
	print('Model loaded from %s \n' % filename)

	accuracy, gap, result = compute_accuracy_and_GAP(net, testloader, return_result = True)
	print('Accuracy of the network on the test images: %.3f' % accuracy)
	print('GAP of the network on the test images: %.3f' % gap)
	result.to_csv("../Doc/Models/" + DENSENET + ".csv" )
        
for VGG in VGGS:
	if VGG [:5] == "Vgg16" or VGG [:5] == "vgg16":
		MODEL = torchvision.models.vgg16
	else:
		MODEL = torchvision.models.vgg19
	filename = "../Doc/Models/"+ VGG + ".pth"

	net = MODEL(pretrained = PRETRAINED)
	net.classifier[6] = nn.Linear(net.classifier[6].in_features, 47)
	net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
	net.to(device)
	print('Model loaded from %s \n' % filename)

	accuracy, gap, result = compute_accuracy_and_GAP(net, testloader, return_result = True)
	print('Accuracy of the network on the test images: %.3f' % accuracy)
	print('GAP of the network on the test images: %.3f' % gap)
	result.to_csv("../Doc/Models/" + VGG + ".csv" )


