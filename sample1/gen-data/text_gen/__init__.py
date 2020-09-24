# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:27:19 2019

@author: Lycoris radiata
"""
#%%
import random
#%%
    
def gen(chars, string = ''):
    while (chars !=0 ):
        flag = random.random()
        # 0~9
        if flag > 0.5:
            string += chr(random.randint(48,57))
            chars -= 1
            if chars ==0:
                break
            #　special character
            specflag = 0
            if specflag > 0.95:
                if random.random() > 0.5:
                    string += '/'
                    chars -= 1
                else:
                       string += '#' 
                       chars -= 1
            if chars ==0:
                break
        # A~Z
        else:
            assert flag <= 0.5, "flag not >0.5 or <=0.5"
            string += chr(random.randint(65,90))
            chars -= 1
            if chars ==0:
                break
            #　special character
            specflag = 0
            if specflag > 0.85:
                string += '_'
                chars -= 1
                if chars ==0:
                    break
    return(string)