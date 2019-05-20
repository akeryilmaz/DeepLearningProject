import matplotlib.pyplot as plt

def extract_loss(path):
	file = open(path, "r")
	losses = []
	split = file.read().split("loss:")
	for element in split[1:]:
		losses.append(element[1:6])
	return losses

def extract_accurasies(path):
	file = open(path, "r")
	accuracies = []
	split = file.read().split("images:")
	for element in split[1:]:
		accuracies.append(element[1:6])
	return accuracies

resnet18CleanData = extract_loss("../Doc/Logs/resnet18SplitData.txt")
resnet18NoisyData = extract_loss("../Doc/Logs/resnet18SplitWithNonlandmarkData.txt")

densenet121CleanData = extract_loss("../Doc/Logs/densenet121SplitData.txt")
densenet121NoisyData = extract_loss("../Doc/Logs/densenet121SplitDataWithNonlandmark.txt")

plt.plot(resnet18CleanData, label="Resnet18 On Clean Data", linestyle='dashed', color='red')
plt.plot(resnet18NoisyData, label="Resnet18", color='red')

plt.plot(densenet121CleanData, label="Densenet121 On Clean Data", linestyle='dashed', color='blue')
plt.plot(densenet121NoisyData, label="Densenet121", color='blue')

plt.title('\"Clean\" Data Experiments')
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.show()
