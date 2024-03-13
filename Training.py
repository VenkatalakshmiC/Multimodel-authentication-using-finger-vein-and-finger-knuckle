# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:01:04 2023

@author: MANJU
"""

import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import colorsys
from feature_extraction import lbp_feature
from feature_extraction import hog_feature
from scipy.io import savemat

warnings.filterwarnings("ignore")
path = os. getcwd()
Data_path = (path + '\\' + 'knuckle')
folders = os.listdir(Data_path)
nfolders = len(folders)
nsamples = 7
feature_matrix_knuckle = []
feature_knuckle_label = []
k=0
for i in range(nfolders-3):
    folderName=folders[i]
    imgFilepth = (Data_path + '\\' + folderName) 
    files = os.listdir(imgFilepth)
    numFiles = len(files)
    
    
    for j in range(nsamples):
        print(j)
        fileName = files[j]
        imgFileName = (imgFilepth + '\\' + fileName)
        img= cv2.imread(imgFileName)
        knuckle_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        [seg_image,features] = lbp_feature(knuckle_img)
        feature_matrix_knuckle.insert(k,features)
        feature_knuckle_label.insert(k,i+1)
        # plt.figure(1)
        # plt.imshow(seg_image)
        # plt.show()
        k = k+1
        fig, axs = plt.subplots(2)
        axs[0].imshow(knuckle_img)
        axs[1].imshow(seg_image)
        plt.show()
        
#####################################################################       
path = os. getcwd()
Data_path = (path + '\\' + 'vein')
folders = os.listdir(Data_path)
nfolders = len(folders)
nsamples = 7
feature_matrix_vein =[]
feature_vein_label = []
k=0

for i in range(nfolders-4):
    folderName=folders[i]
    imgFilepth = (Data_path + '\\' + folderName) 
    files = os.listdir(imgFilepth)
    numFiles = len(files)
    print(i)
    
    for j in range(nsamples):
        fileName = files[j]
        imgFileName = (imgFilepth + '\\' + fileName)
        img= cv2.imread(imgFileName)
        vein_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        [seg_image,features] = hog_feature(vein_img)
        feature_matrix_vein.insert(k,features)
        feature_vein_label.insert(k,i+1)
        fig, axs = plt.subplots(2)
        axs[0].imshow(vein_img)
        axs[1].imshow(seg_image)
        plt.show()
        #feature_vectors = feature_extraction(img)
        k = k+1 

savemat('Train_data_knukle.mat', mdict={'features_knuckle': feature_matrix_knuckle,'labels_knuckle':feature_knuckle_label}) 
savemat('Train_data_vein.mat', mdict={'features_vein': feature_matrix_vein,'labels_vein':feature_vein_label}) 



   
    