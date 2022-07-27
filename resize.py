from PIL import Image
import os, sys

path = ('C:/Users/Danie/Documents/ArtGAN/train')

def resize():
    for item in os.listdir(path):
        if os.path.isfile(item):
            im = Image.open(item)
            f, e = os.path.splitext(item)
            imResize = im.resize((64,64), Image.ANTIALIAS)
            imResize.save('/trainre'+ f + '.jpg')

resize()

# from PIL import Image
# img = Image.open('/your iamge path/image.jpg') # image extension *.png,*.jpg
# new_width  = 128
# new_height = 128
# img = img.resize((new_width, new_height), Image.ANTIALIAS)
# img.save('/new directory path/output image name.png')