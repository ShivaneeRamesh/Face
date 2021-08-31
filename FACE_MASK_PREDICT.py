from tensorflow import keras
import numpy  as np
import cv2



face_cascade=cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
dataset=np.load("Datasets/MASK.npy",allow_pickle=True)
classes=["Mask","Nomask"]
test_inputs=[]
test_targets=[]
pause=True
video=cv2.VideoCapture(0)

while True:
    ret,frame=video.read()
    faces=face_cascade.detectMultiScale(frame) #x,y,w,h
    for (x,y,w,h) in faces:
        roi=frame[y:y+h,x:x+w].copy()
       
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        image=cv2.resize(roi,(50,50))
        break
    cv2.imshow("frame",frame)
    test_inputs.append(image)
    test_inputs=np.array(test_inputs)
    normalised_test_inputs=test_inputs/255
    model=keras.models.load_model("Models/bestface.hdf5")
    for i,test in enumerate(normalised_test_inputs):
      #loop to print prediction for all test images
           prediction=model.predict_classes(test.reshape(1,50,50,3))
           cv2.imshow("Image",test)
           print("Prediction",classes[prediction[0][0]])
           if classes[prediction[0][0]] == "Nomask" :
               cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
               cv2.putText(frame,classes[prediction[0][0]] ,(x,y),fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255),thickness=2, lineType=cv2.LINE_AA)
           else :
               cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
               cv2.putText(frame,classes[prediction[0][0]] ,(x,y),fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0),thickness=2, lineType=cv2.LINE_AA)
           
           
           cv2.imshow("Frame",frame)
    
    if cv2.waitKey(0):
        break
video.release()
cv2.destroyAllWindows()
