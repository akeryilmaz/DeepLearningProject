import os
import pandas as pd

binary_file = open("../Doc/class_info_keras_classifier.csv", "r")
categories_file = open("../Keras-VGG16-places365/categories_places365.txt", "r")


data = pd.DataFrame(columns = ["name", "index", "iflandmark"])
while True:
	line = categories_file.readline().split(" ")
	if line[0] == "":
		break
	name = line[0]
	index = line[1].replace("\n", "")
	iflandmark = binary_file.readline().split(" ")[1].replace("\n", "")
	temp = pd.DataFrame({"name":[name], "index":[index], "iflandmark":[iflandmark]})
	data = data.append(temp, ignore_index = True)
data.to_csv("../Doc/iflandmark.csv")
	
