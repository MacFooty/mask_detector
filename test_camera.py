import os
from darkflow.net.build import TFNet
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

model=load_model('mask_classifier.h5')

options = {"model": "cfg/yolo_face.cfg", 
           "load": -1, 
           "threshold": 0.1, 
           "gpu": 0.0}
           
tfnet = TFNet(options)

def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.35:
            face=original_img[top_y:btm_y,top_x:btm_x].copy()
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(face)
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            mask,no_mask=model.predict(face)[0]
            if(mask>no_mask):
                text='mask'
                color=(255,0,0)
            else:
                text='no mask'
                color=(0,0,255)
            print(mask,no_mask)
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), color, 3)
            newImage = cv2.putText(newImage, text, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
            
    return newImage
#tfnet.train()
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == True:
        frame = np.asarray(frame)
        results = tfnet.return_predict(frame)

        new_frame = boxing(frame, results)
        # Display the resulting frame
        cv2.imshow('frame',new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()