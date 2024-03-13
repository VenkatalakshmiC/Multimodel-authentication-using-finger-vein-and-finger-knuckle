# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:02:26 2023

@author: MANJU
"""

import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
from feature_extraction import lbp_feature
from feature_extraction import hog_feature
from scipy.io import loadmat
from tkinter import filedialog
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
file_path = filedialog.askopenfilename()

img= cv2.imread(file_path)
knuckle_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
[seg_image_knuckle_test,features_knuckle_test] = lbp_feature(knuckle_img)
features_knuckle_test = np.transpose(features_knuckle_test)

mat_contents = loadmat('Train_data_knukle.mat')
feature_vectors = mat_contents['features_knuckle']
feature_vectors = feature_vectors[:,:,0]
feature_labels = mat_contents['labels_knuckle']
feature_labels = np.transpose(feature_labels)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(feature_vectors,feature_labels) 
 
knuckle_output= knn.predict(features_knuckle_test) 
print('knuckle image processing completed')
##########################################################################

file_path = filedialog.askopenfilename()

img= cv2.imread(file_path)
vein_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
vein_img = cv2.resize(vein_img, (140,100), interpolation=cv2.INTER_LINEAR)
[seg_image_vein_test,features_vein_test] = hog_feature(vein_img)
features_vein_test = np.transpose(features_vein_test)

mat_contents = loadmat('Train_data_vein.mat')
feature_vectors = mat_contents['features_vein']
# feature_vectors = feature_vectors[:,:,0]
feature_labels = mat_contents['labels_vein']
feature_labels = np.transpose(feature_labels)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(feature_vectors,feature_labels) 
features_vein_test = features_vein_test[np.newaxis,:]
 
vein_output= knn.predict(features_vein_test) 
print('vein image processing completed')

if(knuckle_output==vein_output):
    if(knuckle_output ==1):
        print('Recognized Person 1: Kaviraj')
    elif(knuckle_output ==2):
        print('Recognized Person 2: Lakshmi')
    elif(knuckle_output ==3):
        print('Recognized Person 3: Megha')
    elif(knuckle_output ==4):
        print('Recognized Person 4: Nishu')       
else:
    print('Unknown person! Both biometrics are not matching')






















