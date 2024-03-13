# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:20:30 2023

@author: Meghana R
"""
import matplotlib.pyplot as plt
import numpy as np

threshold= np.arange(0, 0.1, 0.01)
far_knuckle = [0,0,0,33.3333,33.3333,33.3333,33.3333,33.3333,33.3333,33.3333]
frr_knuckle = [75,25,0,0,0,0,0,0,0,0]
tsr_knuckle = 100-np.array(frr_knuckle)
plt.plot(threshold,far_knuckle,threshold,frr_knuckle,threshold,tsr_knuckle)
plt.xlabel('Threshold values')
plt.ylabel('FAR-FRR')
plt.title('FAR-FRR vs threshold for knuckle')
plt.grid()
plt.legend(['FAR','FRR','TSR'],loc='lower right')
plt.show()

threshold= np.arange(0, 1, 0.1)
far_vein = [0,0,0,0,0,25,25,50,50,50]
frr_vein = [75,50,50,50,25,0,0,0,0,0]

tsr_vein = 100-np.array(frr_vein)
plt.plot(threshold,far_vein,threshold,frr_vein,threshold,tsr_vein)
plt.xlabel('Threshold values')
plt.ylabel('FAR-FRR')
plt.title('FAR-FRR vs threshold for vein')
plt.grid()
plt.legend(['FAR','FRR','TSR'],loc='lower right')
plt.show()






