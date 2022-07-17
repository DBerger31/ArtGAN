import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LeakyReLU, BatchNormalization
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

#reshape data to fit model
X_train = x_train.reshape(60000,28,28,1)
X_test = x_test.reshape(10000,28,28,1)

from keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

class Discriminator():
    def __init__(self):
        # super(self).__init__
        self.model = keras.Sequential([
            Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(28,28,2)),
            LeakyReLU(alpha=0.01),
            Conv2D(64, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.01),
            Conv2D(128, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.01),
            Flatten(),
            Dense(1, activation='sigmoid'),]
        )
        # self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1))
        # self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')
        # self.connection = tf.keras.layers.Flatten()
        # self.dense = tf.keras.layers.Dense(10, activation='softmax')
        # self.opt = keras.optimizers.Adam(learning_rate=0.01)

    def compile_discriminator(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self):
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

cnn = Discriminator()
cnn.compile_discriminator()
cnn.train()


#train the model
# cnn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs =3)

