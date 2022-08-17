"""This file resizes all images from a folder to a specific dimension"""

import PIL
import os
import os.path
from PIL import Image

# FIX OSError: image file is truncated (80 bytes not processed)
# AppData\Local\Programs\Python\Python39\lib\site-packages\PIL\ImageFile.py
# Image.LOAD_TRUNCATED_IMAGES = True

# FIX ERROR: Pillow in Python won't let me open image ("exceeds limit")
PIL.Image.MAX_IMAGE_PIXELS = 933120000

f = r'train'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    # FIX OSError: cannot write mode RGBA as JPEG
    img = img.convert('RGB')
    if img.size != (64, 64):
        img = img.resize((64,64))
        img.save(f_img)

