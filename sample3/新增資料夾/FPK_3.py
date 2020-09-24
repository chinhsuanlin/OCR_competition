# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:22:58 2020

@author: user
"""
#https://kknews.cc/zh-tw/code/q55qm6y.html
#https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
import cv2 
import pytesseract 
import numpy as np 
import argparse
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default = './FPK_03_clean.png')
args = vars(ap.parse_args())
# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image)
plt.imshow(gray)
image_vse = 255-image
cv2.imwrite("Image_bin.jpg",image_vse)
kernel_length = np.array(image).shape[1]//80
verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_temp1 = cv2.erode(image_vse, verticle_kernel, iterations=3)
verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
cv2.imwrite("verticle_lines.jpg",verticle_lines_img)
img_temp2 = cv2.erode(image_vse, hori_kernel, iterations=3)
horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)
alpha = 0.5
beta = 1.0 - alpha
img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
img_final_bin = cv2.cvtColor(img_final_bin, cv2.COLOR_BGR2GRAY)
(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("img_final_bin.jpg",img_final_bin)

_, contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
(contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

#%%
img = image.copy()
anchor_x, anchor_y, anchor_w, anchor_h = cv2.boundingRect(contours[1])
anchor_x += 4
anchor_w -= 8
anchor_y += 4
anchor_h -= 8
#cv2.rectangle(img, (anchor_x, anchor_y), (anchor_x+anchor_w, anchor_y+anchor_h), (0, 255, 0), 1)

#del contours[1]
for c in contours[2:]:
# Returns the location and width,height for every contour
    x, y, w, h = cv2.boundingRect(c)
    img[y-5 : y-2, x: anchor_x + anchor_w] = 0 
    img[anchor_y : anchor_y + anchor_h, x-6: x-2] = 0 
#    img[y, x: anchor_x + anchor_w] = 0 
#    img[anchor_y : anchor_y + anchor_h, x] = 0 
    #x, y, w, h = cv2.boundingRect(c)
    #new_img = image[y:y+h, x:x+w]
#    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 10)
#plt.imshow(img)
cv2.imwrite('result.png', img)
#%%
'''    
    
    if (w > 80 and h > 20) and w > 3*h:

        idx += 1

    new_img = image[y:y+h, x:x+w]

    cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)

# If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.

    if (w > 80 and h > 20) and w > 3*h:

        idx += 1

    new_img = image[y:y+h, x:x+w]

    cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)

'''
