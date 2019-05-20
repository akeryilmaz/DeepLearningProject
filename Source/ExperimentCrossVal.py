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

from ExperimentSuite import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
																		std=[0.229, 0.224, 0.225])
TRAINTRANSFORMS = transforms.Compose([transforms.Resize(224),
																		transforms.RandomHorizontalFlip(),
																		transforms.ToTensor(),
																		normalize,
																		])
TESTTRANSFORMS = transforms.Compose([transforms.Resize(224),
																			transforms.ToTensor(),
																			normalize,
																			])


k = 5

experiment = ExperimentSuite(256, TRAINTRANSFORMS, TESTTRANSFORMS, nn.CrossEntropyLoss, optim.Adam, logName = "DensenetCrossVal", device = torch.device('cuda:0'))

experiment.load_train('../Data/threshold')
experiment.load_test('../Data/threshold')

experiment.shuffle_data_indices_k('../Data/threshold', k)

accuracies = []
gaps = []
for i in range (k):
	net = torchvision.models.resnet18(pretrained = False)
	net.fc = nn.Linear(net.fc.in_features, 47)

	trainloader, testloader = experiment.load_split_train_test_kfold(i)

	_, accuracy, gap = experiment.training(net, 0.01, MultiStepLR, 10, trainloader, testloader, do_save = "No", fileName = "", printFreq = 50)
	accuracies.append(accuracy)
	gaps.append(gap)

print("Average accuracy is ", sum(accuracies)/k)
print("Average GAP is ", sum(gaps)/k)
	
