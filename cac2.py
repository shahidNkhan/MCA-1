#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
import random
from sklearn.datasets import load_sample_image
import warnings; warnings.simplefilter('ignore')  # Fix NumPy issues.
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs

# import cv2


# In[28]:


def index_distance(i1,j1,i2,j2,H,W):
    if i1<0 or i1>=H or j1<0 or j1>=W or i2<0 or i2>=H or j2<0 or j2>=W:
        return -1
    x = max(abs(i2-i1),abs(j2-j1))
#     if x == 7:
#         print(x, "---", i1,"-",j1,"--", i2,"-",j2)
    if x == 1:
        return 0
    if x == 3:
        return 1
    if x == 5:
        return 2
    if x == 7:
        return 3
    return -1


# In[43]:


def create_correlogram(img,M,D):
    
    #resize with M, (768/1024)*M
    img = img.resize((64,48),Image.ANTIALIAS)
    array = np.asarray(img.convert('LA'))[:,:,0]//4
    acc = np.zeros((M,D))
    arr = array
    num = np.zeros((M,1))
    counter = 0
    
#     print("mapping from image to 64 colours complete")
    counter = 0
    for i1 in range(arr.shape[0]):
        for j1 in range(arr.shape[1]):
            counter += 1
#             print("counter:",counter)
            num[arr[i1,j1],0] += 1
            for i2 in range(i1-8,i1+8):
                for j2 in range(j1-8,j1+8):
                    d = index_distance(i1,j1,i2,j2,arr.shape[0],arr.shape[1])
                    if d == -1:
                        continue
                    else:
#                         denom[arr[i1,j1],d] += 1
                        if arr[i1,j1] == arr[i2,j2]:
                            acc[arr[i1,j1],d] += 1
    for i in range(M):
        for j in range(D):
            if acc[i,j] == 0:
                continue
            acc[i,j] = acc[i,j]/(num[i,0]*(8*(j*2+1)))
    return acc


# In[49]:


def compare_cac(acgram1,acgram2,M,D):
    ans = 0
    ans_arr = np.abs(acgram1-acgram2)/(1+acgram1+acgram2)
    ans = ans_arr.sum()/M
    return ans


# In[50]:


image_path = "HW-1/images"
query_path = "HW-1/train/query"
ground_truth_path = "HW-1/train/ground_truth"
image_names = [f for f in listdir(image_path) if isfile(join(image_path, f))]
query_names = [f for f in listdir(query_path) if isfile(join(query_path, f))]
ground_truth_names = [f for f in listdir(ground_truth_path) if isfile(join(ground_truth_path, f))]


# In[51]:


img1 = Image.open('HW-1/images/all_souls_000013.jpg')
acorrelogram1 = create_correlogram(img1,64,4)
print("1 down")
img2 = Image.open('HW-1/images/all_souls_000091.jpg')
acorrelogram2 = create_correlogram(img2,64,4)
print("2 down")
img3 = Image.open('HW-1/images/all_souls_000220.jpg')
acorrelogram3 = create_correlogram(img3,64,4)

img4 = Image.open('HW-1/images/oxford_003521.jpg')
acorrelogram4 = create_correlogram(img4,64,4)

img5 = Image.open('HW-1/images/cornmarket_000071.jpg')
acorrelogram5 = create_correlogram(img5,64,4)

img6 = Image.open('HW-1/images/hertford_000105.jpg')
acorrelogram6 = create_correlogram(img6,64,4)


# In[52]:


print(compare_cac(acorrelogram1,acorrelogram1,64,4))
print(compare_cac(acorrelogram1,acorrelogram2,64,4))
print(compare_cac(acorrelogram1,acorrelogram3,64,4))
print(compare_cac(acorrelogram1,acorrelogram4,64,4))
print(compare_cac(acorrelogram1,acorrelogram5,64,4))
print(compare_cac(acorrelogram1,acorrelogram6,64,4))


# In[56]:


img1 = Image.open('HW-1/images/all_souls_000013.jpg')
acorrelogram1 = create_correlogram(img1,64,4)

fmaxval = 5
smaxval = 5
tmaxval = 5

fmaxe = ''
smaxe = ''
tmaxe = ''

counter = 0

for imn in image_names:
    if imn == 'all_souls_000013.jpg': continue
    pth = 'HW-1/images/'+imn
    imgx = Image.open(pth)
    acorrelogramx = create_correlogram(imgx,64,4)
    val = compare_cac(acorrelogram1,acorrelogramx,64,4)
    if val < fmaxval:
        tmaxval  = smaxval
        smaxval = fmaxval
        fmaxval = val
        
        tmaxe = smaxe
        smaxe = fmaxe
        fmaxe = pth
        
    elif val < smaxval:
        tmaxval = smaxval
        smaxval = val
        
        tmaxe = smaxe
        smaxe = pth
        
    elif val < tmaxval:
        tmaxval = val
        
        tmaxe = pth
        
#     print(counter)
    if counter%100==0:
        print("\n\n",counter,fmaxe,smaxe,tmaxe,"\n\n")
    counter+=1


# In[ ]:


img1 = Image.open('HW-1/images/oxford_001216.jpg') 
plt.imshow(img) 
print(img.format) 
print(img.mode)


# In[5]:


img2 = Image.open('HW-1/images/christ_church_000884.jpg')
img2


# In[21]:


img2 = img2.resize((64,48),Image.ANTIALIAS)
np.asarray(img2.convert('LA'))[:,:,0]


# In[ ]:


npimg_recolored2,recolored_centroids2,centroids2 = quantize_npimg(np.asarray(img2),64)
acorrelogram2 = create_correlogram(npimg_recolored2,centroids2,64,4)


# In[14]:


t = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,15,17,200]])


# In[26]:


(1+t).sum()


# In[ ]:




