#https://holypython.com/how-to-batch-resize-multiple-images-in-python-via-pil-library/#:~:text=You%20can%20resize%20multiple%20images,os%20(operating%20system)%20library.&text=By%20using%20os.,file%20names%20in%20a%20directory.&text=After%20that%2C%20all%20you%20have,each%20image%20in%20the%20directory.
import PIL
import os
import os.path
from PIL import Image

# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
# FIX OSError: image file is truncated (80 bytes not processed)
# AppData\Local\Programs\Python\Python39\lib\site-packages\PIL\ImageFile.py
# Image.LOAD_TRUNCATED_IMAGES = True

# https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
# FIX ERROR: Pillow in Python won't let me open image ("exceeds limit")
PIL.Image.MAX_IMAGE_PIXELS = 933120000

f = r'train'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    # https://stackoverflow.com/questions/48248405/cannot-write-mode-rgba-as-jpeg
    # FIX OSError: cannot write mode RGBA as JPEG
    img = img.convert('RGB')
    if img.size != (64, 64):
        img = img.resize((64,64))
        img.save(f_img)

