# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:35:05 2019

@author: Lycoris radiata
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:31:42 2019

@author: Lycoris radiata
"""
#%%
from PIL import Image, ImageDraw, ImageFont
import text_gen
import random
import numpy as np
import cv2 as cv2
import os
#%% Paramters
#row = 130
#col = 130
ch = 1


art = 'Library/Fonts/arial.ttf'
width = 130
height_range = [35, 90, 120, 150, 180, 210]
#scale = row/height
font_size = 28
steps = [-3, -2, -1, 0, 1, 2]
y_compact = [-15, -10, -5, 0, 5, 10, 15]
#steps = [-3]
caves = os.listdir('caves/')
with open ('annotation.txt', 'w'):
    pass
with open ('lexicon.txt', 'w'):
    pass
#%%
for i in range(90000):
    height = random.choice(height_range)
    seed = random.randint(2,5)
    string = text_gen.gen(seed)
    # Special label  a == "/", b = "\"
    label = string
    string = string.replace("a", "/").replace("b", "\\")
    boxes = [0]*len(string)

    #%%
    cave = random.randint(0,3)
    im = Image.new("L", (width, height), "#fff") 
    flag = 0
    d = ImageDraw.Draw(im)
    x, y = 15, (height/2 - font_size/2) #The  Anchor of the fucking text
    xx = 3
    flag_bar = 0
    falg_ul = 0

    #%% 
    if falg_ul > 0.95:
        pirror = random.randint(1,len(string))
        last = random.randint( pirror,len(string))
        if random.random() > 0.95 and abs(pirror-last)>0 :
            # bar
            boxes[pirror:last] = [1]*abs(pirror-last)
            flag = 1
        elif abs(pirror-last)>0:    
        #Lower
            boxes[pirror:last] = [-1]*abs(pirror-last)
            flag = -1
    #%% Draw
    y = (height/2 - font_size/2)
    for idx, val in enumerate(boxes):
        if val == 0:
            # Normal

            fill = '#000'
            font = ImageFont.truetype(art, font_size) # select the fucking art and size
            d.text((x, y), string[idx], font=font, fill = fill) # Draw labels and black color
            w = d.textsize(string[idx], font=font)[0] # Get fucking pirror of width
            x += w + random.choice(steps) # Anchor + width + Steps
            #y += font.getoffset(string[idx])[1] # Get fucking offset
#            else:
#                if idx <= 8:
#                    y = 10
#                    if cave == 0:
#                        fill = '#fff'
#                    else:
#                        fill = '#000'
#                    font = ImageFont.truetype(art, font_size) # select the fucking art and size
#                    d.text((x, y), string[idx], font=font, fill = fill ) # Draw labels and black color
#                    w = d.textsize(string[idx], font=font)[0] # Get fucking pirror of width
#                    x += w + steps # Anchor + width + Steps
#                    #y += font.getoffset(string[idx])[1] # Get fucking offset
#                else:
#                    y = 25
#                    if cave == 0:
#                        fill = '#fff'
#                    else:
#                        fill = '#000'
#                    font = ImageFont.truetype(art, font_size) # select the fucking art and size
#                    d.text((xx, y), string[idx], font=font, fill = fill) # Draw labels and black color
#                    w = d.textsize(string[idx], font=font)[0] # Get fucking pirror of width
#                    xx += w + steps # Anchor + width + Steps
#                    #y += font.getoffset(string[idx])[1] # Get fucking offset
        elif val == 1:
            # Upper
            y = height/2-font_size/2
            font = ImageFont.truetype(art, font_size)
            bars = '-'
            d.text((x, y), bars, font=font, fill="#000") # Draw labels and black color
            w = d.textsize(bars, font=font)[0] # Get fucking pirror of width
            x += w + random.choice(steps) # Anchor + width + Steps
            #y += font.getoffset(string[idx])[1] # Get fucking offset
        elif val == -1:
            # Lower
            y = height/2-font_size/2
            font = ImageFont.truetype(art, int(font_size*0.8))
            h = d.textsize(string[idx], font=font)[1] # Get fucking pirror of height
            y += font_size/3.5
            d.text((x, y), string[idx], font=font, fill="#000")
            w = d.textsize(string[idx], font=font)[0]
            x += w + random.choice(steps) # Anchor + width + Steps
    #%% -  # 
    if flag_bar > 0.95:
        font = ImageFont.truetype(art, font_size)
        counts = len(string)
        bars = '_'*(counts+1)
        h = d.textsize(bars, font=font)[1] # Get fucking pirror of height
        if flag == 0:
            ratio = 0.7
        elif flag == 1:
            ratio = 1
        elif flag == -1:
            ratio = 0.7
        x, y = 3, (height/2 - font_size*ratio - h)
        d.text((x, y), bars, font=font, fill="#000")
        d.text((x, y+1), bars, font=font, fill="#000")
    #%% Result
    if flag_bar > 0.95:
        label = label + 'b'
    #%%
#    im = cv2.resize(im, (31,31))
#    im = cv2.circle(im,(22, 22), 22, (0, 0, 0), 1)
#    im = cv2.resize(im, (200,31))
#    res = ''
#    for index, v in enumerate(boxes):
#        if v == 1:
#            res +='-'
#        else:
#            res += label[index]
#    res = res.replace('-', 'c')
#    cave = np.ones((row, col, ch))
#    cave *=255
#    anchor = int ((row - height)/2), int((col - width)/2)
#    y, x = anchor
#    cave[y:y+height, x:x+width, : ] = im
#    cave = cave.astype('uint8')
    #%%
    im = im.resize((130, 150),Image.ANTIALIAS)
    code = str(random.randint(0,10000))
    cv2.imwrite( './imgs/' + label  + '_height' + str(height) + '_' + str(code) + '.png' , np.array(im))
    with open ('annotation.txt', 'a') as f:
        f.write('./imgs/' + label  + '_height' + str(height) + '_' + str(code) + '.png'  +' ' + str(i) + '\n')
    with open ('lexicon.txt', 'a') as f:
        f.write(label + '\n')
"""
原始碼
"""
    #%% 上標
    #font = ImageFont.truetype(art, int(font_size*2/3))
    #h = d.textsize("T/M", font=font)[1] # Get fucking pirror of height
    #y -= h / 2
    #d.text((x, y), "T/M", font=font, fill="#000")
    #%% 下標
    #font = ImageFont.truetype(art, int(font_size*2/3))
    #h = d.textsize("T/M", font=font)[1] # Get fucking pirror of height
    #y += h / 1.3
    #d.text((x, y), "T/M", font=font, fill="#000")
    #%% 本體
    #font = ImageFont.truetype(art, font_size) # select the fucking art and size
    #d.text((x, y), string, font=font, fill="#000") # Draw labels and black color
    #w = d.textsize(string, font=font)[0] # Get fucking pirror of width
    #x += w # Anchor + width
    #y += font.getoffset(string)[1] # Get fucking offset