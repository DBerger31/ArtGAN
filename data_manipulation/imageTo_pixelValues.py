"""This file creates images.npy and labels.txt from the Kaggle dataset to be used in the model"""

import PIL
import os
import os.path
from PIL import Image
import numpy as np
import pandas as pd

import time 
start_time = time.time()

# FIX OSError: image file is truncated (80 bytes not processed)
# AppData\Local\Programs\Python\Python39\lib\site-packages\PIL\ImageFile.py
# Image.LOAD_TRUNCATED_IMAGES = True

# FIX ERROR: Pillow in Python won't let me open image ("exceeds limit")
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# Making images.npy
# Convert each image to pixel values and put it in an array
r"""
f = r'5_Genres'
array = []
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    array.append(np.array(img))
array = np.stack(array, axis = 0)

# Save 4D array to a npy file
np.save('images(128x128).npy', array)

# Double check whether the first element in the numpy array in images.npy is equal to the first image numpy array
d = np.load('images(128x128).npy')
print(d.shape)
print(d[0])
"""
######################################################################################
# Making labels.txt
# Match filename with it's correct labels and put all labels to an array
# Then save it as txt

f = r'5_Genres'
filename = []
for file in os.listdir(f):
    filename.append(file)

df = pd.read_csv(r'train_info.csv')
label = []
for name in filename:
    x = df.loc[df['filename'] == name, 'genre'].iloc[0]
    label.append(x)
print(label)

# label each genre with a number
label = [0 if i =='abstract' 
        else 1 if i == 'landscape'
        else 2 if i =='religious painting'
        else 3 if i =='portrait'
        else 4 if i =='sketch and study'
        else i for i in label]
print(label)

# Save 1D array to a txt file
np.savetxt('5_Genres_Labels.txt', label, fmt='%d')

d = np.loadtxt('5_Genres_Labels.txt', dtype=int)
print(d.shape)

# Print program runtime
elapsed = (time.time() - start_time)
time = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
print("Process finished --- %s --- " % (time))
