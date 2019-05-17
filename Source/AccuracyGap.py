# Select data directory
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
from torch.optim.lr_scheduler import MultiStepLR
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from vgg16_places_365 import *
BATCHSIZE = 1

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TESTTRANSFORMS = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize,])																
inverse = transforms.Compose([transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0/0.229, 1.0/0.224, 1.0/0.225]), transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0])])
																			
PRINT_FREQ = 200

MODEL = torchvision.models.resnet18

RESNETS = ["resnet18SplitData","resnet18SplitWithNonlandmarkData"]

if os.path.isdir('../Data/splitWithNonlandmark/test'):
	data_dir = '../Data/splitWithNonlandmark/test'
else:
	raise Exception("Data directry not found!")

class_information = pd.read_csv('../Doc/class_info_keras_classifier.csv')
nonlandmark_dir = '../Data/splitData/nonlandmarkTrain'

# Places365 Model
model = VGG16_Places365(weights='places')
data = pd.read_csv('../Doc/iflandmark.csv')

# Select the device for training
device = torch.device('cuda:0')
#device = torch.device('cpu')

def check_nonlandmark(image):
        image_array = inverse(image[0].cpu()).permute(1,2,0)
        image_array = image_array.numpy()
        image_array_reshaped = np.expand_dims(image_array, 0)
        predictions = model.predict(image_array_reshaped)
        probs = np.sort(predictions)
        classes = np.argsort(predictions)
        classes = classes.tolist()[0]
        probs = probs.tolist()[0]
        nonlandmarkProb = 0.0
        landmarkProb = 0.0
        while landmarkProb < 0.15 :
                currentClass = classes.pop()
                iflandmark = data[data.index == currentClass].iflandmark.get_values()[0]
                if iflandmark == 1:
	                nonlandmarkProb += probs.pop()
                else:
	                landmarkProb += probs.pop()
                if nonlandmarkProb >= 0.85:
                        return True
        return False
        
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
			ifnonlandmark = check_nonlandmark(images)
			confidence, predicted = torch.max(outputs.data, 1)
			if(ifnonlandmark):
			        confidence = confidence*0.1
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

        # TODO Check the below code :)
        net = MODEL()
        net.fc = nn.Linear(net.fc.in_features, 47)
        net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        net.to(device)
        print('Model loaded from %s \n' % filename)

        accuracy, gap, result = compute_accuracy_and_GAP(net, testloader, return_result = True)
        print('Accuracy of the network on the test images: %.3f' % accuracy)
        print('GAP of the network on the test images: %.3f' % gap)
        result.to_csv("../Doc/Models/Conf" + RESNET + ".csv" )
        







