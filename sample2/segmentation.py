# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:38:56 2019

@author: Lycoris radiata
"""
# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import statistics as stat
import os
from math import cos as cos
from math import sin as sin
#%%
def adjust_gamma(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)  

#%%
def seg():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default = './FPK_06.jpg')
    args = vars(ap.parse_args())
    
    # load the image, clone it for output, and then convert it to grayscale
    image = cv2.imread(args["image"])
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.0, gray.shape[0]/35, minRadius = 10, maxRadius = 30, param1=50,param2=30)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    	# show the output image
#        cv2.imshow("output", np.hstack([image, output]))
#        cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    for circle in circles:
#        x,y,r = circle
#        for rad in range(1, 628, 1):
#            rad = rad / 50
#            yy = int(y - round(r*sin(rad),2))
#            xx = int(x + round(r*cos(rad),2))
#            image[yy,xx, :] = 255
#            image[yy-1,xx+1, :] = 255
#            image[yy+1,xx-1, :] = 255
    cv2.imwrite('output.jpg', output)
    """
    INIT
    """
    y_sort = sorted(range(len(circles[:,1])), key=lambda k: circles[:,1][k])
    x_sort = sorted(range(len(circles[:,0])), key=lambda k: circles[:,0][k])
    ROWS = []
    circles_y = circles[y_sort]
    circles_x = circles[x_sort]
    anchor_x, anchor_y =  circles_x[0,0], circles_y[0,1]
    
    """
    Detect how much rows
    """
    while( len(circles_y) != 0 ):
        tmp = []
        level_y = circles_y[0,1]
        tmp.append(circles_y[0])
        circles_y = np.delete(circles_y, 0, 0)
        breakpoint = 0
        while (breakpoint == 0):
            if circles_y[0,1] - level_y < 35 :
                tmp.append(circles_y[0])
                circles_y = np.delete(circles_y, 0, 0)
                if len(circles_y) != 0:
                    continue
                else:
                    ROWS.append(tmp)
                    breakpoint = 1
            else:
                ROWS.append(tmp)
                breakpoint = 1
    """
    Detect how much cols
    """
    row = 44  
    col = 44
    ch  = 3
    count = 0
    level = np.array(48, dtype = 'int')
    ROW_NAMES = np.r_[65:91]
    ignore = [73, 79, 81, 83, 88, 90]   
    
    """
    ROW_NAMES
    """
    # Encoding
    ROW_NAMES = np.array(list(set(ROW_NAMES) - set(ignore)), dtype='int')
    if len(ROW_NAMES) < len(ROWS):
        compensation = len(ROW_NAMES) - len(ROWS)
        tmp = np.r_[65: 65+abs(compensation)]
        tmp +=90
        ROW_NAMES = np.append(ROW_NAMES, tmp)
    # Decoding
    row_names = np.array([0]*len(ROW_NAMES), dtype = '<U8')
    for ior, vor in enumerate(ROW_NAMES):
        if vor <= 90:
            row_names[ior] = chr(vor)
        else:
            vor -= 90
            row_names[ior] = ('A' + chr(vor))
    """
    Creat COL_NAMES
    """
    for rows, samples in enumerate(ROWS):
        samples = np.array(samples)
        samples = samples[sorted(range(len(samples[:,0])), key=lambda k: samples[:,0][k])]
        col_name = np.empty(0, dtype = 'int')
        name = 1
        for l in range(len(samples)):
            if l == len(samples):
                name += 1
                col_name = np.append(col_name, name)
                break
            elif ( l == 0 ):
                level_x = samples[l,0]
                if level_x - anchor_x < 48:
                    col_name = np.append(col_name, name)
                    name +=1
                else:
                    steps = np.array( np.around( (level_x-anchor_x) / level) ).astype('int')
#                    steps = stat.median(steps)
                    name += steps
                    col_name = np.append(col_name, name)
                    name += 1
            else:
                level_x = samples[l,0]
                level_p = samples[l-1,0]
                if (level_x - level_p) < 48:
                    col_name = np.append(col_name, name)
                    name += 1
                else:
                    steps = np.array( np.around((level_x - level_p)/level)).astype('int')
#                    steps = stat.median(steps)
                    name += steps -1
                    col_name = np.append(col_name, name)
                    name += 1
        for c, circle in enumerate(samples) :
            x, y, r = circle
            img = image[int(y-1*r):int(y + 1*r ), int(x - 1*r): int(x + 1*r)]
#            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height = img.shape[0]
            width = img.shape[1]
            cave = np.ones((row, col, ch))
            cave *=255
            anchor = int ((row - height)/2), int((col - width)/2)
            y, x = anchor
            cave[y:y+height, x:x+width, : ] = img
            cave = cave.astype('uint8')
            cave = cv2.cvtColor(cave, cv2.COLOR_RGB2GRAY)
#            cave[cave >= 180] =102
#            cave = cv2.adaptiveThreshold(cave,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2)
#            cave = cv2.resize(cave, (200,31))
#            cave = cv2.cvtColor(cave, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('./rawdata/' + row_names[rows] + str(col_name[c])+ '.png', cave)
#            cv2.imwrite(path + row_name + col_name  + '.png', cave)
            count +=1
def segg():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default = './FPK_02.jpg')
    args = vars(ap.parse_args())
    
    # load the image, clone it for output, and then convert it to grayscale
    image = cv2.imread(args["image"])
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.0, gray.shape[0]/35, minRadius = 20, maxRadius = 25, param1=50,param2=30)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    	# show the output image
#        cv2.imshow("output", np.hstack([image, output]))
#        cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    for circle in circles:
#        x,y,r = circle
#        for rad in range(1, 628, 1):
#            rad = rad / 50
#            yy = int(y - round(r*sin(rad),2))
#            xx = int(x + round(r*cos(rad),2))
#            image[yy,xx, :] = 255
#            image[yy-1,xx+1, :] = 255
#            image[yy+1,xx-1, :] = 255
    cv2.imwrite('output.jpg', output)
    """
    INIT
    """
    y_sort = sorted(range(len(circles[:,1])), key=lambda k: circles[:,1][k])
    x_sort = sorted(range(len(circles[:,0])), key=lambda k: circles[:,0][k])
    ROWS = []
    circles_y = circles[y_sort]
    circles_x = circles[x_sort]
    anchor_x, anchor_y =  circles_x[0,0], circles_y[0,1]
    
    """
    Detect how much rows
    """
    while( len(circles_y) != 0 ):
        tmp = []
        level_y = circles_y[0,1]
        tmp.append(circles_y[0])
        circles_y = np.delete(circles_y, 0, 0)
        breakpoint = 0
        while (breakpoint == 0):
            if circles_y[0,1] - level_y < 35 :
                tmp.append(circles_y[0])
                circles_y = np.delete(circles_y, 0, 0)
                if len(circles_y) != 0:
                    continue
                else:
                    ROWS.append(tmp)
                    breakpoint = 1
            else:
                ROWS.append(tmp)
                breakpoint = 1
    """
    Detect how much cols
    """
    row = 144  
    col = 35
    ch  = 3
    count = 0
    level = np.array(140, dtype = 'int')
    ROW_NAMES = np.r_[65:76]
    ROW_NAMES = np.arange(75,64,-1)
    ignore = [73, 79, 81, 83, 88, 90]   
    
    """
    ROW_NAMES
    """
    # Encoding
    ROW_NAMES = np.array(list(set(ROW_NAMES) - set(ignore)), dtype='int')
    # Inverse
    ROW_NAMES = ROW_NAMES[::-1]
    if len(ROW_NAMES) < len(ROWS):
        compensation = len(ROW_NAMES) - len(ROWS)
        tmp = np.r_[65: 65+abs(compensation)]
        tmp +=90
        ROW_NAMES = np.append(ROW_NAMES, tmp)
    # Decoding
    row_names = np.array([0]*len(ROW_NAMES), dtype = '<U8')
    for ior, vor in enumerate(ROW_NAMES):
        if vor <= 90:
            row_names[ior] = chr(vor)
        else:
            vor -= 90
            row_names[ior] = ('A' + chr(vor))
    """
    Creat COL_NAMES
    """
    for rows, samples in enumerate(ROWS):
        samples = np.array(samples)
        samples = samples[sorted(range(len(samples[:,0])), key=lambda k: samples[:,0][k])]
        col_name = np.empty(0, dtype = 'int')
        name = 1
        for l in range(len(samples)):
            if l == len(samples):
                name += 1
                col_name = np.append(col_name, name)
                break
            elif ( l == 0 ):
                level_x = samples[l,0]
                if level_x - anchor_x < level:
                    col_name = np.append(col_name, name)
                    name +=1
                else:
                    steps = np.array( np.around( (level_x-anchor_x) / level) ).astype('int')
#                    steps = stat.median(steps)
                    name += steps
                    col_name = np.append(col_name, name)
                    name += 1
            else:
                level_x = samples[l,0]
                level_p = samples[l-1,0]
                if (level_x - level_p) < level:
                    col_name = np.append(col_name, name)
                    name += 1
                else:
                    steps = np.array( np.around((level_x - level_p)/level)).astype('int')
#                    steps = stat.median(steps)
                    name += steps -1
                    col_name = np.append(col_name, name)
                    name += 1
        # Inverse col_name
        col_name = col_name[::-1]
        for c, circle in enumerate(samples) :
            x, y, r = circle
            img = image[int(y + 50  - 0.65*r):int(y + 50  + 0.8*r), int(x - 3.0*r): int(x + 3.0*r)]
            img = cv2.resize(img, (144,35))
#            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#            height = img.shape[0]
#            width = img.shape[1]
#            cave = np.ones((row, col, ch))
#            cave *=255
#            anchor = int ((row - height)/2), int((col - width)/2)
#            y, x = anchor
#            cave[y:y+height, x:x+width, : ] = img
#            cave = cave.astype('uint8')
#            cave = cv2.cvtColor(cave, cv2.COLOR_RGB2GRAY)
#            cave[cave >= 180] =102
#            cave = cv2.adaptiveThreshold(cave,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2)
#            cave = cv2.resize(cave, (200,31))
#            cave = cv2.cvtColor(cave, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('./test/' + row_names[rows] + str(col_name[c])+ '.png', img)
#            cv2.imwrite(path + row_name + col_name  + '.png', cave)
            count +=1                 
#seg()
#segg()
#gen_test()