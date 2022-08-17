"""Load dataset into keras and displaying images from the batches"""

from subprocess import SubprocessError
from turtle import color
import numpy as np
from sklearn.utils import shuffle 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Counting the number of files in a folder
dataset_path = "./Genres/Abstract"
count = 0
for path in os.listdir(dataset_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dataset_path, path)):
        count += 1
print('File count:', count)

dataset_path =  "Genres/"
image_size = (64,64)
batch_size = 10

train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)

# Training set 
train_batches = train_datagen.flow_from_directory(
    dataset_path,
    target_size = image_size,
    batch_size = batch_size, 
    class_mode = "categorical",
    shuffle = True,
    subset = "training"

)
# Validation set
validation_batches = train_datagen.flow_from_directory(
    dataset_path,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = "categorical",
    subset = "validation"
)
# Testing set
test_batches = train_datagen.flow_from_directory(
    dataset_path,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = True,
    subset = "validation"
)

# Plot the first 5 images in the batch
def plotImages(images_arr):
    fig, axes = plt.subplots(1,5)
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

print(train_batches.class_indices)
imgs, labels = train_batches[0]
print(imgs.shape)
print(labels.shape)
print(labels[:5])
plotImages(imgs)