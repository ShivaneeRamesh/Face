Face_Mask_Detection:
Real time face-mask detection using Deep Learning and OpenCV

About Project:
This project uses a Deep Neural Network, more specifically a Convolutional Neural Network, to differentiate between images of people with and without masks.
The CNN managesd to get an accuracy of 99.7% on the training dataset and 98.5% on the testing dataset. Then the stored weights of this CNN are used to classify 
as mask or no mask, in real time, using OpenCV. With the webcam capturing the video, the frames are preprocessed and and fed to the model to accomplish this task.

Libraries used: Tensorflow, Keras, OpenCV, Numpy.

Dataset:
The training and testing dataset contain 2000 images for mask and no mask respectively.
