# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:30:43 2020

@author: Lycoris radiata
"""

import cv2
import os
path = 'test/'
files = os.listdir(path)

for i in files:
    img = cv2.imread(path + i)
    img = cv2.resize(img, (130,150))
    cv2.imwrite(path + i, img)