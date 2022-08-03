#https://holypython.com/how-to-batch-resize-multiple-images-in-python-via-pil-library/#:~:text=You%20can%20resize%20multiple%20images,os%20(operating%20system)%20library.&text=By%20using%20os.,file%20names%20in%20a%20directory.&text=After%20that%2C%20all%20you%20have,each%20image%20in%20the%20directory.
import PIL
import os
import os.path
from PIL import Image
import numpy as np
import pandas as pd

import time 
start_time = time.time()

# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
# FIX OSError: image file is truncated (80 bytes not processed)
# AppData\Local\Programs\Python\Python39\lib\site-packages\PIL\ImageFile.py
# Image.LOAD_TRUNCATED_IMAGES = True

# https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
# FIX ERROR: Pillow in Python won't let me open image ("exceeds limit")
PIL.Image.MAX_IMAGE_PIXELS = 933120000



# Convert each image to pixel values and put it in an array
r"""
f = r'top10_train'
array = []
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    array.append(np.array(img))

array = np.stack(array, axis = 0)
print(array)
print(array.shape)
print(array[0])
print(array[0].shape)
"""
# https://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly
# Save 4D array to a npy file
#np.save('images.npy', array)
d = np.load('images.npy')
print(d.shape)
print(d[0])
img = Image.open('10.jpg')
x = np.array(img)
print("10.jpg matrix")
print(x)
print(d[0] == x)

# Match filename with it's correct labels and put all labels to an array
# Then save it as txt
r"""
f = r'top10_train'
array = []
filename = []
for file in os.listdir(f):
    filename.append(file)

print(filename)

# https://stackoverflow.com/questions/36684013/extract-column-value-based-on-another-column-pandas-dataframe
df = pd.read_csv(r'trainnew.csv')
label = []
for name in filename:
    x = df.loc[df['filename'] == name, 'genre'].iloc[0]
    label.append(x)
print(label)

# Save 1D array to a txt file
np.savetxt('labels.txt', label, fmt='%d')
"""
f = r'top10_train'
filename = []
for file in os.listdir(f):
    filename.append(file)
print(filename)

d = np.loadtxt('labels.txt', dtype=int)
print(d)
print(d.shape)


"""
img = Image.open('1.jpg')
print(np.array(img))
print(np.array(img).shape)
print(np.array(img).flatten().shape)

img = Image.open('1482.png')
print(np.array(img))
print(np.array(img).shape)
print(np.array(img).flatten())
print(np.array(img).flatten().shape)
"""
# Print program runtime
elapsed = (time.time() - start_time)
time = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
print("Process finished --- %s --- " % (time))