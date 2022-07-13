import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
import numpy as np
# Testing Dataset
from keras.datasets import mnist

# Constants and hyperparameters
batch_size = 64
num_of_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128

#download mnist data and split into train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])

class Discriminator(keras.Sequential):
    def __init__(self):
        self.conv1 = tf.keras.layers.Conv2D()
        self.conv2 = tf.keras.layers.Conv2D()
        self.connection = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense()


