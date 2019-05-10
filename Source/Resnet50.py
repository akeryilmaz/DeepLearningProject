# Select data directory
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

#Set this to True if model is already trained
skip_training = False

if os.path.isdir('../Data/threshold'):
	data_dir = '../Data/threshold'
else:
	#TODO Change the error given here.
	raise Exception("Data directry not found!")

print('The data directory is %s' % data_dir)

# Select the device for training
device = torch.device('cuda:0')
#device = torch.device('cpu')

if skip_training:
  # The models are always evaluated on CPU
  device = torch.device("cpu")

#TODO: Test below function

def load_split_train_test(datadir, valid_size = .2):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
																		std=[0.229, 0.224, 0.225])

	train_transforms = transforms.Compose([transforms.Resize(224),
																		transforms.ToTensor(),
																		normalize,
																		])

	test_transforms = transforms.Compose([transforms.Resize(224),
																			transforms.ToTensor(),
																			normalize,
																			])

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
									sampler=train_sampler, batch_size=32)
	testloader = torch.utils.data.DataLoader(test_data,
									sampler=test_sampler, batch_size=32)
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

net = torchvision.models.resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 47)
net.to(device)

#Training settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
n_epochs = 10

print("Starting Training.")
#Training Loop
net.train()
for epoch in range(n_epochs):
	running_loss = 0.0
	print_every = 200  # mini-batches
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
			running_loss = 0.0

		if skip_training:
			break
	if skip_training:
		break

	# Print accuracy after every epoch
	accuracy = compute_accuracy(net, testloader)
	print('Accuracy of the network on the test images: %.3f' % accuracy)

print('Finished Training')

# Save the network to a file
filename = 'resnet50Pretrained.pth'
if not skip_training:
	try:
		do_save = input('Do you want to save the model (type yes to confirm)? ').lower()
		if do_save == 'yes':
			torch.save(net.state_dict(), filename)
			print('Model saved to %s' % filename)
		else:
			print('Model not saved')
	except:
		  raise Exception('The notebook should be run or validated with skip_training=True.')
else:
	# TODO Check the below code :)
	net = ResNet(n_blocks, n_channels=16)
	net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
	net.to(device)
	print('Model loaded from %s' % filename)

# Compute the accuracy on the test set
accuracy = compute_accuracy(net, testloader)
print('Accuracy of the network on the test images: %.3f' % accuracy)



