import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Testing Dataset
from keras.datasets import mnist
#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])

