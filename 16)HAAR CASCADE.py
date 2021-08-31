#########################################################################################Import essential Libraries
import numpy as np                                                                      						#For working with matrices
import cv2
from tensorflow import keras  #Has functions for image processing
face_cascade = cv2.CascadeClassifier('Cascades\haarcascade_frontalface_default.xml')                		#Creates an object for face detection using the trained xml file
eye_cascade = cv2.CascadeClassifier('Cascades\haarcascade_eye.xml')
model = keras.models.load_model("Datasets/best.hdf5")
#Creates an object for eye detection using the trained xml file
cap = cv2.VideoCapture(0)                                                               					#Creates an object to obtain images from laptop webcam
while 1:                                                                                						#Infinite loop until broken by keyboard interrupt
    ret, img = cap.read()                                                               						#Obtain image from camera
    faces = face_cascade.detectMultiScale(img)                                         					#Obtain a list of coordinates of rectangles covering the faces
    for (x,y,w,h) in faces:
                                                                                                                                 #For each rectangle covering the faces in the image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,[classes], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)                                        #Draw a green rectanlge using the coordinates
        roi_gray = img[y:y+h, x:x+w]                                                   						#Cut the region of interest from the whole image
        roi_color = img[y:y+h, x:x+w]                                                   						#Obtain the colored region of interest
        eyes = model.detectMultiScale(roi_gray)                                   					#Obtain a list of coordinates of rectangles covering the eyes
        for (ex,ey,ew,eh) in eyes:                                                      						#For each rectangle covering the eyes in the image
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)                  				#Draw the rectangle
	   
	    #Display the final image
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff                                                          						#Keyboard interrupt
    if k == 27:                                                                         						#If escape key is pressed all windows gets closed
        break                                                                           						#
											#
cap.release()                                                                           						#
cv2.destroyAllWindows()                                                                 						#
#########################################################################################
