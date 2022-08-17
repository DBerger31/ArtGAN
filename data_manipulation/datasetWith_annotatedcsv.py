"""This file reads in the dataset with annotated csv"""

import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('trainnew.csv')
file_paths = df['filename'].values
labels = df['genre'].values
ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))

def read_image(image_file, label):
    image = tf.io.read_file(r"top10_train" + "/" + image_file)
    image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
    return image, label

ds_train = ds_train.map(read_image)

for x, y in ds_train:
    print(x)
