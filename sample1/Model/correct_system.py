# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:20:19 2019

@author: Lycoris radiata
"""

import cv2 as cv2
import matplotlib.pyplot as plt

img = cv2.imread('test/A1.png')
plt.imshow(img)

with open ('eval/000/A1.txt', 'r') as f:
    label = f.readline()