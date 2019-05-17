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
from NetworkLoader import *
from NonlandmarkModel import *

def load_test(datadir):
  test_transforms = TESTTRANSFORMS
  test_data = torchvision.datasets.ImageFolder(datadir,
					        transform=test_transforms)
  testloader = torch.utils.data.DataLoader(test_data,
					        shuffle=True, batch_size=BATCHSIZE)
  return testloader

def compute_accuracy_and_GAP(net, testloader, return_result):
	correct = 0
	total = 0
	result = pd.DataFrame(columns = ["pred", "conf", "true"])
	with torch.no_grad():
		for i, (images, labels) in enumerate(testloader):
			print(i)
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			confidence, predicted = net.get_prediction(images)
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

if __name__ == "__main__":
	BATCHSIZE = 256

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
																			std=[0.229, 0.224, 0.225])
	TESTTRANSFORMS = transforms.Compose([transforms.Resize(224),
																				transforms.ToTensor(),
																				normalize,
																				])

	MODELS = ["densenet121SplitData", "densenet121SplitDataWithNonlandmark"]

	DEVICE = torch.device('cuda:0')
	#DEVICE = torch.device('cpu')

	DATA_DIR = '../Data/splitDataNonlandmark/test'

	testloader = load_test(DATA_DIR)

	for modelname in MODELS:
		net = NonlandmarkModel(modelname, DEVICE)
		accuracy, gap, result = compute_accuracy_and_GAP(net, testloader, return_result = True)
		print('Accuracy of the network on the test images: %.3f' % accuracy)
		print('GAP of the network on the test images: %.3f' % gap)
		result.to_csv("../Doc/Results/" + modelname + "NonlandmarkModelTest.csv" )
  


