# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 23:11:36 2020

@author: user
"""

import numpy as np
import argparse
import cv2
import pandas as pd
from skimage import measure

def seg():

    flg=1#轉成Dataframe
    imgflg =0#畫圖用
    def img2gray(img_):#將圖像轉灰階，並以黑色為1，白色為0輸出
        #RGB轉灰階
        gray = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
        #消除雜訊及背景
        gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        #閥值二值化
        _,gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        #壓制0~1間
        gray = gray/255
        #0、1反轉
        gray = 1 - gray
        #轉回uint8
        gray = gray.astype('uint8')
        return gray
    def gray2label(img_,connectivity_=2,area_ = 2000):#給予滿足的label，需要減1滿足和properties一樣的index
        #計算label數
        labels = measure.label(img_,connectivity=connectivity_)
        #label性質
        properties = measure.regionprops(labels)
        #取閥值
        valid_label = np.empty(0)
        #留下選取的範圍
        for prop in properties:       
            if prop.area > area_ :
                valid_label = np.append(valid_label, int(prop.label))
        valid_label = list(map(int, valid_label))
        return valid_label,properties
    
    
    class info:
        def __init__(self,img,properties):#圖像與性質資訊
            self.img=img
            self.properties=properties
            self.minr_=[];self.minc_=[];self.maxr_=[]
            self.maxc_=[];self.w_=[];self.h_=[]
            self.centroid_=[]
        def imginfo(self,lab_length,flg=flg,imgflg=imgflg,plusc=0,plusr=0):#plusc,plusr移位用
            self.position_=[]#個別存成一個dict
            for lab in lab_length:#要剪1
                (minr,minc,maxr,maxc)=self.properties[lab].bbox#取邊界
                centroid = self.properties[lab].centroid#取中心
                
                dict_num = {'minr':minr+plusr,'minc':minc+plusc,'maxr':maxr+plusr,'maxc':maxc+plusc,'w':(maxc-minc),'h':(maxr-minr),'centroid':centroid}#紀錄邊界資訊
                self.position_.append(dict_num)
                if imgflg ==1:#畫圖
                    self.img = cv2.rectangle( self.img,   (minc+plusc,minr+plusr), (maxc+plusc,maxr+plusr),(0, 255, 0), 4  )
                    #self.img = cv2.circle(self.img,(int(centroid[1]+plusc),int(centroid[0]+plusr)),2,(255,0,0),5)
            
                if flg == 1:
                    self.minr_.append(minr+plusr)
                    self.minc_.append(minc+plusc)
                    self.maxr_.append(maxr+plusr)
                    self.maxc_.append(maxc+plusc)
                    self.w_.append(maxc-minc)
                    self.h_.append(maxr-minr)
                    cent_0=centroid[0]+plusr
                    cent_1=centroid[1]+plusc
                    centroid=(cent_0,cent_1)
                    self.centroid_.append(centroid)
        def todf(self,flg=flg):
            self.df = pd.DataFrame(data={'minr':self.minr_,'minc':self.minc_,'maxr':self.maxr_,'maxc':self.maxc_,'w':self.w_,'h':self.h_,'centroid':self.centroid_})
            return self.df
    
    
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default = './FPK_05.jpg')
    args = vars(ap.parse_args())
        
    # load the image, clone it for output, and then convert it to grayscale
    img = cv2.imread(args["image"])
    img_gray = cv2.imread(args["image"], 0)
    output = img.copy()
    #plt.imshow(img)
    gray = img2gray(img)
    gray = cv2.medianBlur(gray, 5)
    #plt.imshow(gray)
    
    
    valid_label,properties = gray2label(gray)
    
    imgbox = info(gray,properties)
    imgbox.imginfo( np.array(valid_label)-1)#要剪1，轉型態
    imgboxdf =imgbox.todf(flg=1)
    
    ##取上半部
    imgup = gray[0:135-5,:]#裁切圖、使用灰階圖找資訊
    labels=measure.label(imgup,connectivity=2)#尋找label
    properties = measure.regionprops(labels)#label的性質
    imgupbox = info(img,properties)
    imgupbox.imginfo(range(len(properties)))
    imgupboxdf =imgupbox.todf(flg=1)
    ##取左半部
    imgleft = gray[:,0:147-5]
    labels=measure.label(imgleft,connectivity=2)#尋找label
    properties = measure.regionprops(labels)#label的性質
    imgleftbox = info(img,properties)
    imgleftbox.imginfo(range(len(properties)))
    imgleftboxdf =imgleftbox.todf(flg=1)
    #plt.imshow(img)
    
    inchr={1:'A',2:'B',3:'C',4:'D',
           5:'E',6:'F',7:'G',8:'H',
           9:'J',10:'K',11:'L',12:'M',
           13:'N',14:'P',15:'R',16:'T',
           17:'U',18:'V',19:'W',20:'Y'}
    ##截圖
    for up_index_,up_value_ in enumerate(imgupboxdf.centroid):
        for lt_index_,lt_value_ in enumerate(imgleftboxdf.centroid):
            #名稱標記
            if up_index_ >= 3 and up_index_ <= 5:
                continue
            else:
                noation = inchr[lt_index_+1] + str(up_index_+1)
                #中心點
                seg_cen = (int(lt_value_[0]),int(up_value_[1]))
                #gray = cv2.rectangle( img,(seg_cen[1]-60,seg_cen[0]-60), (seg_cen[1]+60,seg_cen[0]+60),(0, 255, 0), 6)
                saveimg = img_gray[seg_cen[0]-25:seg_cen[0]+45,seg_cen[1]-60:seg_cen[1]+60]
    #                saveimg = cv2.medianBlur(saveimg, 5)
                cv2.imwrite( './test/' + (noation+'.png'), saveimg)
    
seg()
