######################################IMPORTING ESSENTIAL LIBRARIES######################################
import numpy as np 												#Python's vector/matrix processing library.
import cv2 												#Image processing library.
import os 													#Library for file path handling.
import random 												#To work with random operations.
##########################################DATASET PREPARATION##########################################
directory = "FaceImages"											#Destination directory which contains images.
emotions = ["Mask","Nomask"] 										#These folders will be accessed by the program for data.
train_data = [] 												#Empty array to hold the emotion data.
for emotion in emotions: 											#First loop to select the folder.
	path = directory+"/"+emotion 										#Append the destination directory with the folder name.
	index = emotions.index(emotion) 										#Get the index value of the current emotion. This is the target value.
	for img in os.listdir(path): 										#Second loop to select the image.
		try:											#Some images might not load properly.
			img_array = cv2.imread(path+"/"+img)							#Load the image into the program as a numpy array.
			new_array = cv2.resize(img_array, (50,50)) 						#Resize the image to reduce computational complexity.
			train_data.append([new_array,index]) 							#Attach the current input and target to the dataset.
		except: 											#When an error occurs.
			pass 										#Do nothing and go to the next iteration.
train_data=np.array(train_data)
random.shuffle(train_data) 											#Shuffle the dataset to prevent overfitting while training.
data = train_data[:10]											#Get the first 10 images and labels.
for features,labels in data: 											#Loop through the sample
	print(labels) 											#Dsiplay the shuffled label values.
np.save("Datasets/MASK.npy",train_data) 								#Save the training data in the Datasets folder.
#########################################################################################################
