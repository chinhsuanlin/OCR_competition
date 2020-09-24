# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:17:28 2019

@author: Lycoris radiata
"""
import random
#%%
with open ('annotation.txt', 'r') as f:
    lines = f.readlines()
#%%
random.shuffle(lines)
#%%
breakpoint = int(len(lines)*0.8)
with open ('annotation_train.txt', 'w') as f:
    [ f.write(lines[i]) for i in range (breakpoint)]
    
with open ('annotation_val.txt', 'w') as f:
    [ f.write(lines[i+breakpoint]) for i in range (len(lines) - breakpoint)]