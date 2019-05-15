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

MODEL = torchvision.models.resnet18
FILENAME = 'resnet18Pretrained'


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

def compute_accuracy(net, testloader):
	net.eval()
	correct = 0
	total = 0
	confidences = []
	predictions = []
	true_labels = [] 
	with torch.no_grad():
		for images, labels in testloader:
			images, labels = images.to(device), labels.to(device)
			outputs = net(images)
			confidence, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			confidences.append(confidence)
			predictions.append(predictions)
			true_labels.append(labels)
	return correct / total, np.array(confidences), np.array(predictions), np.array(true_labels)

def GAP_vector(pred, conf, true, return_x=False):
    '''
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition. 
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation

    Returns:
        GAP score
    '''
    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap

testloader = load_test(data_dir)

filename = "../Doc/Models/"+ FILENAME + ".pth"

# TODO Check the below code :)
net = MODEL()
net.fc = nn.Linear(net.fc.in_features, 47)
net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
net.to(device)
print('Model loaded from %s \n' % filename)

accuracy, conf, pred, true = compute_accuracy(net, testloader)
print('Accuracy of the network on the test images: %.3f' % accuracy)

gap, x = GAP_vector(pred, conf, true, return_x = True)

print('GAP of the network on the test images: %.3f' % gap)

print(x)




