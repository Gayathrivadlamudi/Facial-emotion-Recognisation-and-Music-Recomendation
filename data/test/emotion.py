from keras.models import load_model 
from time import sleep 
from keras.preprocessing.image import img_to_array 
from keras.preprocessing import image
import cv2 
import numpy as np
face classifier = cv2.CascadeClassifier('test.xml')
classifier load_model('./train.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
cap = cv2.VideoCapture(0)
while True: 
     # Grab a single frame of video  18  ret, frame cap.read()
     ret,frame = ret, frame = cap.read()
     labels = []
     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     #detect the face
     faces = face classifier.detectMultiScale(gray,1.3,5) 
     for (x,y,w,h) in faces:
        #boundary of face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w] 
        #mobile net arch should be gie input image 48x48 
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation-cv2.INTER_AREA)
        #converting the image into the array 
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi) 
            roi = np.expand_dims (roi, axis=0)  
            # make a prediction  preds = classifier.predict(roi)[0]  
            preds = classifier.predict(roi)[0]
            print("\nprediction", preds)  
            label-class labels [preds.argmax()]  
            print("\nprediction max = ", preds.argmax())  
            print("\nlabel ",label)  label_position= (x,y)  
            cv2.putText(frame, label, label_position,cv2.FONT HERSHEY SIMPLEX, 2, (0,255,0),3)
            else:  
                cv2.putText(frame, 'No Face Found', (20,60), cv2. FONT HERSHEY SIMPLEX ,2,(0,255,0),3)
            print("\n\n")
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitkey(1) & 0*FF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
