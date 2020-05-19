#!/usr/bin/env python
# coding: utf-8

# In[14]:


import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
# img = cv2.imread("sunflower.jpg",0)
from os.path import isfile, join
from scipy import ndimage
from scipy.ndimage import filters
from scipy import spatial
k = sqrt(2)
sigma = 1.0
# img = img/255.0  #image normalization


# In[15]:


#  check the extra time for the float part. if it is too expensive, try without
def create_arrays_hv(n,c):    
    start_lim = int(floor(-n/2))
    end_lim = int(floor(n/2))+1
    x_array = np.arange(start_lim,end_lim)
    shape = x_array.shape[0]
    if(c=='v' or c=='V'):
        if isinstance(n,float):
            return x_array.reshape(shape,1).astype(float)
        else:
            return x_array.reshape(shape,1)
    elif(c=='h' or c=='H'):
        if isinstance(n,float):
            return x_array.reshape(1,shape).astype(float)
        else:
            return x_array.reshape(1,shape)


# In[16]:


def LoG(factor):
    sigma_1 = sigma*factor
    n = np.ceil(sigma_1*6)
    x = create_arrays_hv(n,'H')
    y = create_arrays_hv(n,'V')
    x_term = np.power(x,2)
    y_term = np.power(y,2)
    sigma_squared = sigma_1*sigma_1
    xy1 = (-( x_term + y_term)/(2*sigma_squared))
    e_xy = np.exp(xy1)
    final_filter = (e_xy) * (-(2*sigma_1**2) + (x*x + y*y) ) * (1/(2*np.pi*sigma_1**4))
    return final_filter


# In[17]:


def apply_filter(img,y):
    filter_log = LoG(y)
    image = cv2.filter2D(img,-1,filter_log)
    return image*image


# In[18]:


log_image_np = [0,0,0,0,0,0,0,0,0]
square_scales = [0,1,4,9,16,25,36,49,64]
for i in range(0,9):
    y = np.power(sqrt(2),i) 
#         sigma_1 = sigma*y #sigma 9
    image = apply_filter(img,y)
    log_image_np[i] = image
log_image_np = np.array(log_image_np)


# In[24]:


def detect_blob(log_image_np,threshhold):
    co_ordinates = []
    for i in range(1,img.shape[0]):
        for j in range(1,img.shape[1]):
            slice_img = log_image_np[:,i-1:i+2,j-1:j+2]
            maxi = slice_img.argmax()
            
            result = np.amax(slice_img)
            if result >= threshhold:
                x = (slice_img.argmax()//slice_img.shape[2])%slice_img.shape[1] + i
                x -= 1
                y = slice_img.argmax()%slice_img.shape[2]+j
                y -= 1
                z = (slice_img.argmax()//slice_img.shape[2])//slice_img.shape[1]
                co_ordinates.append((x,y,k**z*sigma))
    return co_ordinates
set_co = set(detect_blob(log_image_np,0.03))
co_ordinates = list(set_co)


# In[23]:


fig, ax = plt.subplots()
nh = img.shape[0]
nw = img.shape[1]
count = 0
plt.imshow(img,cmap="gray") 
for blob in co_ordinates:
    y = blob[0]
    x = blob[1]
    r = blob[2]
    c = plt.Circle((x, y), r*k, color='blue', linewidth=0.4, fill=False)
    plt.gcf().gca().add_artist(c)
plt.plot()  
plt.show()


# In[7]:


image_path = "HW-1/images/"
query_path = "HW-1/train/query/"
ground_truth_path = "HW-1/train/ground_truth/"
output_path = "HW-1/blob_detection/query_keypoints/"
image_names = [f for f in listdir(image_path) if isfile(join(image_path, f))]
query_names = [f for f in listdir(query_path) if isfile(join(query_path, f))]
ground_truth_names = [f for f in listdir(ground_truth_path) if isfile(join(ground_truth_path, f))]


# In[36]:


counter = 0
for q in query_names:
    print(counter)
    counter += 1
    f = open(query_path+q, "r")
    name_img = f.readline().split()[0][5:]
    q_image_name = image_path + name_img +".jpg"
    f.close()
    img = cv2.imread(q_image_name,0)
    scale_percent = 75 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)/255.0
    log_image_np = [0,0,0,0,0,0,0,0,0]
    square_scales = [0,1,4,9,16,25,36,49,64]
    for i in range(0,9):
        y = np.power(sqrt(2),i) 
    #         sigma_1 = sigma*y #sigma 9
        image = apply_filter(img,y)
        log_image_np[i] = image
    log_image_np = np.array(log_image_np)
    setco_i = set(detect_blob(log_image_np,0.003))
    co_ordinates = list(setco_i)
    np.savez(name_img, np.asarray(co_ordinates))
#     break


# In[34]:


a = np.load(q_image_name+".npz") #make sure you use the .npz!
b = a['arr_0']
print(b.shape)


# In[208]:


cv2.imread(q_image_name,0)


# In[203]:


st = 'oxc1_all_souls_000013'


# In[205]:


st[5:]


# In[216]:


np.amax(img)


# In[29]:


len(co_ordinates.shape)


# In[33]:


np.asarray(co_ordinates).shape


# In[ ]:




