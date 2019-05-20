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


experiment = ExperimentSuite(32, TRAINTRANSFORMS, TESTTRANSFORMS, nn.CrossEntropyLoss, optim.Adam, logName = "Resnet50", device = torch.device('cuda:0'))

trainloader, testloader = experiment.load_split_train_test('../Data/threshold', 1/8)

net = torchvision.models.resnet50(pretrained = False)
net.fc = nn.Linear(net.fc.in_features, 47)

_, accuracy, gap = experiment.training(net, 0.01, MultiStepLR, 10, trainloader, testloader, do_save = "yes", fileName = "Resnet50", printFreq = 200)

