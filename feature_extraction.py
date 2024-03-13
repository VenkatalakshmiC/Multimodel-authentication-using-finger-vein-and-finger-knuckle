# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 13:41:01 2023

@author: MANJU
"""
import cv2
import numpy as np
import math
from numpy import linalg as LA
from scipy import ndimage

def lbp_feature(knuckle_img):

    hsv_img = cv2.cvtColor(knuckle_img, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv_img)
    
    # Define thresholds for channel 1 based on histogram settings
    channel1Min = 5
    channel1Max = 50
    
    # Define thresholds for channel 2 based on histogram settings
    channel2Min = 70
    channel2Max = 220
    
    # Create mask based on chosen histogram thresholds
    sliderBW = (h >= channel1Min ) & (h <= channel1Max) & (s >= channel2Min ) & (s <= channel2Max) 
    BW = sliderBW
    BW= ndimage.binary_opening(BW,structure=np.ones((3,3))).astype(int)
    seg_image = np.zeros((knuckle_img.shape[0],knuckle_img.shape[1],3),dtype="uint8")
    seg_image[:,:,0] = np.multiply(knuckle_img[:,:,0],BW)
    seg_image[:,:,1] = np.multiply(knuckle_img[:,:,1],BW)
    seg_image[:,:,2] = np.multiply(knuckle_img[:,:,2],BW)
    
   
    
    [row,col] = np.where(seg_image[:,:,0]!=0)
    mir = np.min(row)
    mar = np.max(row)
    mic = np.min(col)
    mac = np.max(col)
    roi_image = seg_image[mir:mar,mic:mac,:]   
    
    roi_image = cv2.resize(roi_image, (100,140), interpolation=cv2.INTER_LINEAR)
     
    Input_Im = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
    R =1
    L = 2*R+1  #The size of the LBP label
    C = int(np.round(L/2))
    row_max = Input_Im.shape[0]-L+1
    col_max = Input_Im.shape[1]-L+1
    LBP_Im = np.zeros((row_max, col_max), dtype='uint8')
    for i in range(row_max):
        for j in range(col_max):
            A = Input_Im[i : i + L, j : j + L]
            A = A.astype(np.int32)
            A = A-A[C,C]
            A[A>0] = 1
            A[A<=0] = 0
            LBP_Im[i,j] = A[C-1,L-1] + A[L-1,L-1]*2 + A[L-1,C-1]*4 + A[L-1,0]*8 + A[C-1,0]*16 + A[0,0]*32 + A[0,C-1]*64 + A[0,L-1]*128
            
    count = np.zeros((256,1),dtype='double')                   
        
    for k in range (256):
        for i in range (LBP_Im.shape[0]):
            for j in range (LBP_Im.shape[1]):
                if LBP_Im[i,j]==k:
                    count[k,0] = count[k,0]+1
                  
          
       

    return(roi_image,count/max(count))

def hog_feature(vein_img):
    
    
    hsv_img = cv2.cvtColor(vein_img, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv_img)
    
    # Define thresholds for channel 1 based on histogram settings
    channel1Min = 5
    channel1Max = 50
    
    # Define thresholds for channel 2 based on histogram settings
    channel2Min = 70
    channel2Max = 220
    
    # Create mask based on chosen histogram thresholds
    sliderBW = (h >= channel1Min ) & (h <= channel1Max) & (s >= channel2Min ) & (s <= channel2Max) 
    BW = sliderBW
    seg_image = np.zeros((vein_img.shape[0],vein_img.shape[1],3),dtype="uint8")
    seg_image[:,:,0] = np.multiply(vein_img[:,:,0],BW)
    seg_image[:,:,1] = np.multiply(vein_img[:,:,1],BW)
    seg_image[:,:,2] = np.multiply(vein_img[:,:,2],BW)
    
    [row,col] = np.where(seg_image[:,:,0]!=0)
    mir = np.min(row)
    mar = np.max(row)
    mic = np.min(col)
    mac = np.max(col)
    roi_image = seg_image[mic:mac,mir:mar,:]   
    
    roi_image = cv2.resize(roi_image, (100,140), interpolation=cv2.INTER_LINEAR)

    img = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
        
    h, w = img.shape
    # print(img.shape)

    img_copy_one = img.copy()  # IY
    img_copy_two = img.copy()  # IX

    for i in range(w - 2):
        img_copy_two[:, i] = img[:, i] - img[:, i + 2]

    for i in range(h - 2):
        img_copy_one[i, :] = img[i, :] - img[i + 2, :]

    angle = np.arctan((img_copy_two/img_copy_one)) #np.arctan(np.divide(img_copy_two, img_copy_one, out=np.zeros_like(img_copy_two), where=img_copy_one != 0))
    angle = (angle*180.0)/math.pi
    angle = np.add(angle, 90)
    magnitude = np.sqrt((img_copy_one ** 2 + img_copy_two ** 2))

    if np.isinf(angle).any():
        angle[np.isinf(angle)] = 0
    if np.isnan(angle).any():
        angle[np.isnan(angle)] = 0
    if np.isnan(magnitude).any():
        magnitude[np.isnan(magnitude)] = 0


    features = []

    for i in range(int((h / 8)) - 2):
        for j in range(int((w / 8)) - 2):

            mag_patch = magnitude[8 * i: 8 * i + 15, 8 * j: 8 * j + 15]
            ang_patch = angle[8 * i: 8 * i + 15, 8 * j: 8 * j + 15]
            block_feature = []

            for x in range(2):
                for y in range(2):
                    angleA = ang_patch[8 * x:8 * x + 7, 8 * y: 8 * y + 7]
                    magA = mag_patch[8 * x:8 * x + 7, 8 * y: 8 * y + 7]

                    histr = np.zeros(9)

                    for p in range(7):
                        for q in range(7):

                            alpha = angleA[p, q]

                            if 10 < alpha <= 30:
                                histr[0] = histr[0] + magA[p, q] * (30 - alpha) / 20
                                histr[1] = histr[1] + magA[p, q] * (alpha - 10) / 20

                            elif 30 < alpha <= 50:
                                histr[1] = histr[1] + magA[p, q] * (50 - alpha) / 20
                                histr[2] = histr[2] + magA[p, q] * (alpha - 30) / 20

                            elif 50 < alpha <= 70:
                                histr[2] = histr[2] + magA[p, q] * (70 - alpha) / 20
                                histr[3] = histr[3] + magA[p, q] * (alpha - 50) / 20

                            elif 70 < alpha <= 90:
                                histr[3] = histr[3] + magA[p, q] * (90 - alpha) / 20
                                histr[4] = histr[4] + magA[p, q] * (alpha - 70) / 20

                            elif 90 < alpha <= 110:
                                histr[4] = histr[4] + magA[p, q] * (110 - alpha) / 20
                                histr[5] = histr[5] + magA[p, q] * (alpha - 90) / 20

                            elif 110 < alpha <= 130:
                                histr[5] = histr[5] + magA[p, q] * (130 - alpha) / 20
                                histr[6] = histr[6] + magA[p, q] * (alpha - 110) / 20

                            elif 130 < alpha <= 150:
                                histr[6] = histr[6] + magA[p, q] * (150 - alpha) / 20
                                histr[7] = histr[7] + magA[p, q] * (alpha - 130) / 20

                            elif 150 < alpha <= 170:
                                histr[7] = histr[7] + magA[p, q] * (170 - alpha) / 20
                                histr[8] = histr[8] + magA[p, q] * (alpha - 150) / 20

                            elif 170 < alpha <= 180:
                                histr[8] = histr[8] + magA[p, q] * (180 - alpha) / 20
                                histr[0] = histr[0] + magA[p, q] * (alpha - 170) / 20

                            elif 0 < alpha <= 10:
                                histr[0] = histr[0] + magA[p, q] * (10 + alpha) / 20
                                histr[8] = histr[8] + magA[p, q] * (10 - alpha) / 20
                            else:
                                pass


                    block_feature.extend(histr)


            block_feature = (np.asarray(block_feature))/(math.sqrt(LA.norm(block_feature)** 2)+0.01)
            features.extend(block_feature)


    if np.isnan(features).any():
        features[np.isnan(features)] = 0.0

    features = (np.asarray(features))/(math.sqrt(LA.norm(features)** 2)+0.001)
    #for z in range len(features):
    #    if features(z)>0.2:

    return(roi_image,features)


