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
import os 

def back_processing():
    files = os.listdir('test/')
    for i in files:
        img = cv2.imread('test/' + i)
        img = rotate()
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
 
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
 
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    # 返回旋转后的图像
    return rotated

def seg():
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default = './FPK_07.jpg')
    args = vars(ap.parse_args())
    
    # load the image, clone it for output, and then convert it to grayscale
    image = cv2.imread(args["image"])
#    i = rotate(image, -45)
#    plt.imshow(i)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.0, gray.shape[0]/50, minRadius = 150, maxRadius = 180, param1=50,param2=30)
    
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
    
    	 #show the output image
#    	cv2.imshow("output", np.hstack([image, output]))
#    	cv2.waitKey(0)
#    cv2.destroyAllWindows()
    cv2.imwrite('output.png', output)
    path = 'test/'
    
    
    cols = 7
    rows = 7
    seq = []
    for i in range(rows):
        seq.append(i*cols)
    aa=['B','C','D','R','G','F','J']   
    for index, i in enumerate(seq): 
        row_name = chr( (index+65) %90)
        row_name = aa[index]
        y_sort = sorted(range(len(circles[:,1])), key=lambda k: circles[:,1][k])[i:i+cols]
        infos = circles[y_sort]
        infos_sort_x = sorted(range(len(infos[:,0])), key=lambda k: infos[:,0][k])
        
        X = infos[infos_sort_x][:,0]
        Y = infos[infos_sort_x][:,1]
        R = infos[infos_sort_x][:,2]
        # ASCII ID
    #    cols = 49 # 1~9
    #    rows = 65 # A~Z
        count = i
        
        for i in range(len(X)):
            col_name = str( abs( (i % 9) -7  ))
            x = X[i]
            y = Y[i]
            r = R[i]
            img = image[int(y-0.6*r):int(y + 0.6*r), int(x - 0.9*r): int(x + 0.9*r)]
            img = cv2.resize(img, (135,90))
    #        plt.imshow(img)
            cv2.imwrite(path + row_name + col_name  + '.png', img)
            count +=1
seg()
##%%
#    """
#    Creating the fuckly donkey csv
#    """
#
#from pandas import DataFrame
#
#Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
#        'Price': [22000,25000,27000,35000]
#        }
#df = DataFrame(Cars, columns= ['Brand', 'Price'])
#
#print (df)