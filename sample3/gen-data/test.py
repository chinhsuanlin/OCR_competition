# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:16:40 2019

@author: Lycoris radiata
"""

import cv2 as cv2
import os 

caves = os.listdir("caves/")

for i in caves:
    img = cv2.imread("caves/" + i, 0)
    cv2.imwrite("caves/" + i, img)
