# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:16:40 2019

@author: Lycoris radiata
"""

import cv2 as cv2
import matplotlib.pyplot as plt

img = cv2.imread('A3.png', 0)

cave = cv2.imread('caves/2.png',0)
upper = img[0:25, :]
plt.imshow(upper)
cave[0:25,:] = upper
plt.imshow(cave)
lower = img[25:,:]
plt.imshow(lower)
cave[25:,:] = lower
plt.imshow(cave)
