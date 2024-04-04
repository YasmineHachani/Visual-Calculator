import os #se balader dans les fichiers
import numpy as np
import pandas as pd
import cv2
import  torch
import mediapipe as mp

from Models.models2D.ClassificationModel import ClassificationModel2D
from utils import create_df, Text_Window

#Load the model
model = ClassificationModel2D(model="ResNet18", NBClass=13, criterion_name="ce", optimizer="Adam", learning_rate=1e-4, schedule="null")
checkpoint = torch.load("results/VisualCalculator/d0vljpch/checkpoints/epoch_epoch=94_val_accs_val/losses=0.024.ckpt")
model = model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Setup for drawing the hand landmarks later
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3)
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



prediction_array = [-1,-1,-1]

y = 100
x = 25

nb_prev = 0 
prev = 0

res = np.zeros((400,800,3),np.uint8)
res.fill(255) 
cv2.putText(res,"Real time calculator", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0,0,255),thickness = 2)

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        print("No Video in Camera frame")
        continue
        
    
    # First, we preprocess the captured image 
    
    df =create_df(image)
    
    if (create_df(df) == np.zeros(63)).all():     # If there is no hand gesture, pass
        pass

    df = np.reshape(image, (3, image.shape[0],image.shape[1]))
    
    
    my_pred = torch.argmax(model.predict(image))    # predicted value for the gesture 

    proba = np.max(model.predict(df))         # prediction probability  
    proba = "{:.2f}".format(round(np.max(proba)*100, 2))
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    
    for i in range(10):
        if int(my_pred)==i:
            predict=str(i)
            
    if int(my_pred)==10:
        predict = "equal"
        
    if int(my_pred) == 11:
        predict = "minus"
        
    if int(my_pred) == 12:
        predict = "plus"
        
    
    
    # we consider the prediction only if we get the same prediction a certain number of times (here 2)
    if nb_prev > 2:
    
        if float(proba) >= 80:  # we only consider predictions with a proba greater than 80%
            image = cv2.putText(image, predict, (50,50), font, fontScale , color , thickness, cv2.LINE_AA)

            #check whether it is the first operand, the second or the operator

            if prediction_array[0] == -1:  #it is the first operand 
                # check wether predict is an operand or an operator 
                if predict == "equal" or predict == "minus" or predict == "plus":
                    image = cv2.putText(image, "please enter a number", (20,90+30), font, fontScale, color, thickness, cv2.LINE_AA)
                else: # if it is a number, it is ok and we can write it on the window
                    prediction_array[0] = predict
                    string = '{}'.format(prediction_array[0])
                    Text_Window(res,string,x,y)
                    x += 20

            elif prediction_array[1] == -1:  # it is the operator
                operator = ""
                if predict == "plus":
                    operator = "+"
                    prediction_array[1] = predict
                    string = '{}'.format(operator)
                    Text_Window(res,string,x,y)

                    x += 20

                elif predict == "minus":
                    operator = "-"
                    prediction_array[1] = predict
                    string = '{}'.format(operator)
                    Text_Window(res,string,x,y)

                    x += 20

                else: 
                    image = cv2.putText(image, "Please enter an operator", (20,90+30), font, fontScale, color, thickness, cv2.LINE_AA)


            else:    #it is the second operand

                if predict == "equal" or predict == "minus" or predict == "plus":
                    image = cv2.putText(image, "please enter a number", (20,90+30), font, fontScale, color, thickness, cv2.LINE_AA)
                else: 
                    prediction_array[2] = predict
                    operation = prediction_array[0]
                    if prediction_array[1] == "plus":
                        operation = int(prediction_array[0]) + int(prediction_array[2])
                    elif prediction_array[1] == "minus":
                        operation = int(prediction_array[0]) - int(prediction_array[2])
                    string = '{} = {}'.format(prediction_array[2],operation)
                    Text_Window(res,string,x,y)
                    x = 25
                    y += 30
                    prediction_array = [-1,-1,-1] #reset the values of the operands

            
            nb_prev = 0 
            
    # counter for predictions
    if prev == predict:
        nb_prev += 1
    else:
        nb_prev = 0

    prev = predict
    
    cv2.imshow("result",res) #window for prediciton
    cv2.imshow('image',image) #webcam window


    k = cv2.waitKey(30) & 0xff #exit when Esc is pressed
    if k == 27:
        break

cap.release() 
cv2.destroyAllWindows() 