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
from performanceEvaluater import compute_accuracy_and_GAP
from Network import *

class ExperimentSuite:
	def __init__(self, batchSize, trainTransforms, testTransforms, lossType, optimizerType, logName, device):
		self.batchSize = batchSize
		self.trainTransforms = trainTransforms
		self.testTransforms = testTransforms
		self.lossType = lossType
		self.optimizerType = optimizerType
		self.log = open("../Doc/Logs/" + logName + ".txt","w+")
		self.device = device
		
	def __del__(self):
		self.log.close()

	def load_split_train_test(self, datadir, splitRatio):

		train_data = torchvision.datasets.ImageFolder(datadir,
										transform=self.trainTransforms)
		test_data = torchvision.datasets.ImageFolder(datadir,
										transform=self.testTransforms)

		num_train = len(train_data)
		indices = list(range(num_train))
		split = int(np.floor(splitRatio * num_train))
		np.random.shuffle(indices)

		train_idx, test_idx = indices[split:], indices[:split]
		train_sampler = SubsetRandomSampler(train_idx)
		test_sampler = SubsetRandomSampler(test_idx)
		trainloader = torch.utils.data.DataLoader(train_data,
										sampler=train_sampler, batch_size=self.batchSize)
		testloader = torch.utils.data.DataLoader(test_data,
										sampler=test_sampler, batch_size=self.batchSize)
		return trainloader, testloader

	def shuffle_data_indices_k(self, datadir, k):

		num_data = sum(len(files) for _, _, files in os.walk(datadir))
		indices = list(range(num_data))
		split = int(np.floor(num_data/k))
		np.random.shuffle(indices)
		self.chunks = np.array_split(indices, k)

		return

	def load_train(self, datadir):

		self.train_data = torchvision.datasets.ImageFolder(datadir,
										transform=self.trainTransforms)
		
		self.log.write('The data directory for train data is %s' % datadir)
		print('The data directory for train data is %s' % datadir)

		return

	def load_test(self, datadir):

		self.test_data = torchvision.datasets.ImageFolder(datadir,
										transform=self.testTransforms)

		self.log.write('The data directory for test data is %s' % datadir)
		print('The data directory for test data is %s' % datadir)
	
		return

	def load_split_train_test_kfold(self, step):

		test_idx = self.chunks[step].tolist()
		train_idx = np.concatenate(self.chunks[:step] + self.chunks[step+1:]).tolist()

		train_sampler = SubsetRandomSampler(train_idx)
		test_sampler = SubsetRandomSampler(test_idx)
		trainloader = torch.utils.data.DataLoader(self.train_data,
										sampler=train_sampler, batch_size=self.batchSize)
		testloader = torch.utils.data.DataLoader(self.test_data,
										sampler=test_sampler, batch_size=self.batchSize)

		return trainloader, testloader

	def training(self, net, lr, lr_schedulerType, n_epochs, trainloader, testloader, do_save, fileName, printFreq):
		
		net.to(self.device)
		optimizer = self.optimizerType(net.parameters(), lr)
		criterion = self.lossType()
		lr_scheduler = lr_schedulerType(optimizer, [5,8], gamma=0.1, last_epoch=-1)
		print_every = printFreq  # mini-batches
		net.train()

		print("Starting Training.")
		self.log.write("Starting Training. \n")
		#Training Loop
		for epoch in range(n_epochs):
			lr_scheduler.step(epoch)
			running_loss = 0.0
			for i, (inputs, labels) in enumerate(trainloader, 0):
				# Transfer to GPU
				inputs, labels = inputs.to(self.device), labels.to(self.device)

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
					self.log.write('[%d, %5d] loss: %.3f \n' % (epoch+1, i+1, running_loss/print_every))
					running_loss = 0.0

		self.log.write('Finished Training \n')
		print('Finished Training')

		network = Network(net, self.device)

		accuracy, gap = compute_accuracy_and_GAP(network, testloader, return_result = False, self.device)
		print('Accuracy of the network on the test images: %.3f' % accuracy)
		print('GAP of the network on the test images: %.3f' % gap)
		self.log.write('Accuracy of the network on the test images: %.3f' % accuracy)
		self.log.write('GAP of the network on the test images: %.3f' % gap)

		# Save the network to a file
		path = "../Doc/Models/"+ fileName + ".pth"
		if do_save == 'yes':
			torch.save(net.state_dict(), path)
			print('Model saved to %s' % path)
			log.write('Model saved to %s \n' % path)
		else:
			print('Model not saved')
			log.write('Model not saved \n')

		return network, accuracy, gap
		


