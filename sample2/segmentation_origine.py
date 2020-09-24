# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:38:56 2019

@author: Lycoris radiata
"""
# import the necessary packages
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
from math import sqrt as sqrt
import copy

def seg():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default = './FPK_02.jpg')
    args = vars(ap.parse_args())
    
    # load the image, clone it for output, and then convert it to grayscale
    image = cv2.imread(args["image"])
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.0, gray.shape[0]/50, minRadius = 20, maxRadius = 25, param1=50,param2=30)
    
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
#    	cv2.imshow("output", np.hstack([image, output]))
#    	cv2.waitKey(0)
#    cv2.destroyAllWindows()
    cv2.imwrite('output.png', output)
    path = 'test/'
    
    
    cols = 6
    rows = 10
    seq = []
    
    row = 31
    col = 200
    ch = 3
    count = 0
    
    for i in range(rows):
        seq.append(i*cols)
        
    for index, i in enumerate(seq): 
        if (75-index) <= 73:
            row_name = chr(75 - index -1)
        else:
            row_name = chr(75-index)
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
            if row_name == 'F' or row_name == 'E':
                if abs(i-len(X)) <= 4:
                        col_name = str( (abs(i-len(X))-2))
            else:   
                col_name = str(abs(i-len(X)))
            x = X[i]
            y = Y[i] + 50
            r = R[i]
            img = image[int(y - 0.65*r):int(y + 0.8*r), int(x - 3.0*r): int(x + 3.0*r)]
            
#            cave = np.ones((row, col, ch))
#            cave *=255
#            anchor = ( int( (row - img.shape[0])/2 ), int( (col - img.shape[1])/2 ))
#            y, x = anchor
#            cave[y:y+img.shape[0], x:x+img.shape[1], : ] = img
#            cave = cave.astype('uint8')
            
            
            #im = im.resize((200, 31),Image.ANTIALIAS)
#            plt.imshow(cave)
            img = cv2.resize(img, (144,35))
            cv2.imwrite(path + row_name + col_name  + '.png', img)
            count +=1
seg()