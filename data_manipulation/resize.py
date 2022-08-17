"""This file contains a function that resizes the images to prep for use in the model"""

import PIL
import os
import os.path
from PIL import Image

f = ('C:/Users/Danie/Downloads/train/train')
newf = ('C:/Users/Danie/Documents/ArtGAN/trainre')
for file in os.listdir(f):
    f_img = newf+"/"+file
    img = Image.open(f_img)
    img = img.resize((64,64))
    img.save(f_img)