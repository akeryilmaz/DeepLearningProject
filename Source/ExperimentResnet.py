# Select data directory
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

BATCHSIZE = 32
SPLIT_RATIO = 1.0/8.0

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
																		std=[0.229, 0.224, 0.225])
TRAINTRANSFORMS = transforms.Compose([transforms.Resize(224),
																		transforms.ToTensor(),
																		normalize,
																		])
TESTTRANSFORMS = transforms.Compose([transforms.Resize(224),
																			transforms.ToTensor(),
																			normalize,
																			])

MODEL = torchvision.models.resnet50
PRETRAINED = True
LOSS = nn.CrossEntropyLoss
OPTIMIZER = optim.Adam
LEARNINGRATE = 0.01
TRAINING_MILESTONES = [8, 16, 20, 24, 28]
LR_GAMMA = 0.1

NUMEPOCHS = 30
PRINT_FREQ = 200
FILENAME = 'resnet50PretrainedMilestoneEpoch30'

#Set this to True if model is already trained
skip_training = False

log = open("../Doc/" + FILENAME + ".txt","w+")

if os.path.isdir('../Data/threshold'):
	data_dir = '../Data/threshold'
else:
	#TODO Change the error given here.
	raise Exception("Data directry not found!")

log.write('The data directory is %s' % data_dir)
print('The data directory is %s' % data_dir)

# Select the device for training
device = torch.device('cuda:0')
#device = torch.device('cpu')

if skip_training:
  # The models are always evaluated on CPU
  device = torch.device("cpu")

#TODO: Test below function

def load_split_train_test(datadir, valid_size = SPLIT_RATIO):

	train_transforms = TRAINTRANSFORMS

	test_transforms = TESTTRANSFORMS

	train_data = torchvision.datasets.ImageFolder(datadir,
									transform=train_transforms)
	test_data = torchvision.datasets.ImageFolder(datadir,
									transform=test_transforms)

	num_train = len(train_data)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	np.random.shuffle(indices)

	train_idx, test_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	test_sampler = SubsetRandomSampler(test_idx)
	trainloader = torch.utils.data.DataLoader(train_data,
									sampler=train_sampler, batch_size=BATCHSIZE)
	testloader = torch.utils.data.DataLoader(test_data,
									sampler=test_sampler, batch_size=BATCHSIZE)
	return trainloader, testloader


# This function computes the accuracy on the test dataset
def compute_accuracy(net, testloader):
	net.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in testloader:
			images, labels = images.to(device), labels.to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	return correct / total

trainloader, testloader = load_split_train_test(data_dir)

net = MODEL(pretrained = PRETRAINED)
net.fc = nn.Linear(net.fc.in_features, 47)
net.to(device)

#Training settings
criterion = LOSS()
optimizer = OPTIMIZER(net.parameters(), lr=LEARNINGRATE)
lr_scheduler = MultiStepLR(optimizer, TRAINING_MILESTONES, gamma=LR_GAMMA, last_epoch=-1)
n_epochs = NUMEPOCHS


print("Starting Training.")
log.write("Starting Training. \n")
#Training Loop
for epoch in range(n_epochs):
	net.train()
	lr_scheduler.step(epoch)
	running_loss = 0.0
	print_every = PRINT_FREQ  # mini-batches
	for i, (inputs, labels) in enumerate(trainloader, 0):
		# Transfer to GPU
		inputs, labels = inputs.to(device), labels.to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if (i % print_every) == (print_every-1):
			print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/print_every))
			log.write('[%d, %5d] loss: %.3f \n' % (epoch+1, i+1, running_loss/print_every))
			running_loss = 0.0

		if skip_training:
			break
	if skip_training:
		break

	# Print accuracy after every epoch
	accuracy = compute_accuracy(net, testloader)
	print('Accuracy of the network on the test images: %.3f' % accuracy)
	log.write('Accuracy of the network on the test images: %.3f \n' % accuracy)

log.write('Finished Training \n')
print('Finished Training')

# Save the network to a file
filename = "../Doc/"+ FILENAME + ".pth"
if not skip_training:
	try:
		do_save = input('Do you want to save the model (type yes to confirm)? ').lower()
		if do_save == 'yes':
			torch.save(net.state_dict(), filename)
			print('Model saved to %s' % filename)
			log.write('Model saved to %s \n' % filename)
		else:
			print('Model not saved')
			log.write('Model not saved \n')
	except:
		  raise Exception('The notebook should be run or validated with skip_training=True.')
else:
	# TODO Check the below code :)
	net = ResNet(n_blocks, n_channels=16)
	net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
	net.to(device)
	print('Model loaded from %s \n' % filename)

log.close() 



