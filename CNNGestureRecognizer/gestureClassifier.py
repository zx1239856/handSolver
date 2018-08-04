# -*- coding: utf-8 -*-

#%%
import cv2
import numpy as np
import os
import time

import gestureCNN as myNN

minValue = 70

saveImg = False
guessGesture = True
visualize = False

lastgesture = -1

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Which mask mode to use BinaryMask or SkinMask (True|False)
binaryMode = False
counter = 0
# This parameter controls number of image samples to be taken PER gesture
numOfSamples = 301
gestname = ""
path = ""
mod = 0


#%%
def saveROIImg(img):
    global counter, gestname, path, saveImg
    if counter > (numOfSamples - 1):
        # Reset the parameters
        saveImg = False
        gestname = ''
        counter = 0
        return
    
    counter = counter + 1
    name = gestname + str(counter)
    print("Saving img:",name)
    cv2.imwrite(path+name + ".png", img)
    time.sleep(0.04 )


#%%
def skinMask(frame, x0, y0, width, height ):
    global guessGesture, visualize, mod, lastgesture, saveImg
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    roi = cv2.resize(roi,(200,200),interpolation=cv2.INTER_CUBIC)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    #Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    
    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    #blur
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    #cv2.imshow("Blur", mask)
    
    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True:
        retgesture = myNN.guessGesture(mod, res)
        if lastgesture != retgesture :
            lastgesture = retgesture
            print myNN.output[lastgesture]
	    if lastgesture == 3:
                import subprocess
                subprocess.call(["xdotool", "type", ' '])
                print myNN.output[lastgesture] + "= Dino JUMP!"
                time.sleep(0.01 )
            #guessGesture = False
    elif visualize == True:
        layer = int(raw_input("Enter which layer to visualize "))
        cv2.waitKey(0)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False
    
    
    return res


#%%
def binaryMask(frame, x0, y0, width, height ):
    global guessGesture, visualize, mod, lastgesture, saveImg
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    #blur = cv2.bilateralFilter(roi,9,75,75)
   
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ret, res = cv2.threshold(blur, minValue, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True:
        retgesture = myNN.guessGesture(mod, res)
        if lastgesture != retgesture :
            lastgesture = retgesture
            #print lastgesture
            
            ## Checking for only PUNCH gesture here
            ## Run this app in Prediction Mode and keep Chrome browser on focus with Internet Off
            ## And have fun :) with Dino
            if lastgesture == 3:
                import subprocess
                subprocess.call(["xdotool", "type", ' '])
                #jump = ''' osascript -e 'tell application "System Events" to key code 49' '''
                #jump = ''' osascript -e 'tell application "System Events" to key down (49)' '''
                #os.system(jump)
                print myNN.output[lastgesture] + "= Dino JUMP!"

            #time.sleep(0.01 )
            #guessGesture = False
    elif visualize == True:
        layer = int(raw_input("Enter which layer to visualize "))
        cv2.waitKey(1)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False

    return res

#%%
def process(frame,x0,y0,width,height):
    global guessGesture, visualize, mod, gestname, path
    
    #Call CNN model loading callback
    print "Will load default weight file"
    mod = myNN.loadCNN(0)
    masked = skinMask(frame,x0,y0,width,height)
    cv2.imshow('ROI', masked)
