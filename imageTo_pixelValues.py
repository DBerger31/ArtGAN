# Making images.npy and labels.txt
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

f = r'top10_train'
array = []
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    array.append(np.array(img))

array = np.stack(array, axis = 0)

# https://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly
# Save 4D array to a npy file
np.save('images.npy', array)

# Double check whether the first element in the numpy array in images.npy is equal to the first image numpy array
d = np.load('images.npy')
print(d.shape)
print(d[0])
img = Image.open('10.jpg')
x = np.array(img)
print("10.jpg matrix")
print(x)
print(d[0] == x)

# Making labels.txt
# Match filename with it's correct labels and put all labels to an array
# Then save it as txt
r"""
f = r'top10_train'
array = []
filename = []
for file in os.listdir(f):
    filename.append(file)

df = pd.read_csv(r'trainnew.csv')
label = []
for name in filename:
    x = df.loc[df['filename'] == name, 'genre'].iloc[0]
    label.append(x)

# Save 1D array to a txt file
np.savetxt('labels.txt', label, fmt='%d')

f = r'top10_train'
filename = []
for file in os.listdir(f):
    filename.append(file)

d = np.loadtxt('labels.txt', dtype=int)
print(d.shape)
"""

# Print program runtime
elapsed = (time.time() - start_time)
time = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
print("Process finished --- %s --- " % (time))