import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt

# Testing Dataset
from keras.datasets import mnist
#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])
plt.show()

class Discriminator(keras.Sequential):
    def __init__(self):
        keras.layers.InputLayer(28,28)
