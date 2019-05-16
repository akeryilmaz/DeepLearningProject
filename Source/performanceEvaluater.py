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

MODEL = torchvision.models.densenet121
FILENAME = 'densenet121SplitData'


device = torch.device('cuda:0')
#device = torch.device('cpu')

if os.path.isdir('../Data/splitDataNonlandmark/test'):
	data_dir = '../Data/splitDataNonlandmark/test'
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
			print (i)
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
		return accuracy, gap, result
	else:
		return accuracy, gap

testloader = load_test(data_dir)

filename = "../Doc/Models/"+ FILENAME + ".pth"

net = MODEL()
net.classifier = nn.Linear(net.classifier.in_features, 47)
net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
net.to(device)
print('Model loaded from %s \n' % filename)

accuracy, gap, result = compute_accuracy_and_GAP(net, testloader, return_result = True)
print('Accuracy of the network on the test images: %.3f' % accuracy)
print('GAP of the network on the test images: %.3f' % gap)
result.to_csv("../Doc/densenet121SplitDataTestResults.csv")




