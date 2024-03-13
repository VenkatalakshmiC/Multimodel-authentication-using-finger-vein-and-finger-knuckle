# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 23:07:34 2023

@author: MANJU
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 13:13:15 2023

@author: MANJU
"""
import os
from scipy.io import loadmat
import numpy as np
import cv2
import math
from feature_extraction import lbp_feature

mat_contents = loadmat('Train_data_knukle.mat')
feature_vectors = mat_contents['features_knuckle']
feature_vectors = feature_vectors[:,:,0]
count_ed=np.zeros((feature_vectors.shape[0],1),dtype='double')

No_Of_persons = 3
Images_Per_knuckle = 1                             
threshold= np.arange(0, 0.1, 0.01)
FAR=[]
path = os. getcwd()
Data_path = (path + '\\' + 'knuckle')
folders = os.listdir(Data_path)
nfolders = len(folders)
m=0
count_value =[]
indexfolder= 4

for i in range(indexfolder,indexfolder+3,1):
    folderName=folders[i]
    imgFilepth = (Data_path + '\\' + folderName) 
    files = os.listdir(imgFilepth)
    numFiles = len(files)
    
    
    for j in range(8,8+Images_Per_knuckle,1):
        print(j)
        fileName = files[j]
        imgFileName = (imgFilepth + '\\' + fileName)
        img= cv2.imread(imgFileName)
        knuckle_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #knuckle_img = cv2.resize(knuckle_img, (140,100), interpolation=cv2.INTER_LINEAR)
        [seg_image,features] = lbp_feature(knuckle_img)
        
        for jj in range(feature_vectors.shape[0]):
            count_ed[jj]=math.sqrt(np.sum(feature_vectors[jj,:]-np.transpose(features)) ** 2)
        
    count_value.insert(m,np.min(count_ed))
    m=m+1  
        
kk=0
far=np.zeros((threshold.shape[0],1),dtype='double')

for k in threshold:
    match_cnt = 0
    mismatch_cnt = 0   
    m=0
    for pp in range(indexfolder,indexfolder+3,1):
        
        for ii in range(8,8+Images_Per_knuckle,1):
    
            if count_value[m]<=k:
                print('Yes..! U r lucky to be here')
                match_cnt = match_cnt+1
            else:
                print('Sorry...! No matches found')
                mismatch_cnt = mismatch_cnt+1
            m=m+1   

    
    far[kk] = (match_cnt/ (No_Of_persons*Images_Per_knuckle))*100                
                   
    kk=kk+1   

        






        