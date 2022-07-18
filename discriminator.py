import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, LeakyReLU, BatchNormalization, Input, Embedding, Reshape, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
# Testing Dataset
from keras.datasets import mnist

'''
Functional API implementation which specifices the layer(arg1)(arg2)

arg1: input
arg2: output
'''   
# define the standalone discriminator model
def discriminator(img_shape, n_labels):

    label = Input(shape=(1,))
	  # embedded layer for creating a vector of 50 for each label that act as weights for each label
    # 50 being the standard for CGANs
    # TODO: POSSIBLY change the randomness of vector
    label_input = Embedding(n_labels, 50)(label)
	  # creates a number of nodes that is easy to reshape to the desired image dimensions
    n_nodes = img_shape[0] * img_shape[1]
    label_input = Dense(n_nodes)(label_input)
    # reshape to additional channel
    label_input = Reshape((img_shape[0], img_shape[1], 1))(label_input)

    # image input
    img_input = Input(shape=img_shape)
    # combine our two inputs an additional channel with be for the labels
    merge = Concatenate()([img_input, label_input])

    # LAYERS 
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)
    output = Dense(1, activation='sigmoid')(fe)
    model = Model([img_input, label_input], output)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32')
    x_train = (x_train - 127.5) / 127.5
    return [x_train, y_train]


# def load_mnist():
# 	# load dataset
# 	(trainX, trainy), (_, _) = keras.datasets.mnist.load_data()
# 	# expand to 3d, e.g. add channels
# 	X = expand_dims(trainX, axis=-1)
# 	# convert from ints to floats
# 	X = X.astype('float32')
# 	# scale from [0,255] to [-1,1]
# 	X = (X - 127.5) / 127.5
# 	return [X, trainy]

x_train,y_train= load_mnist()
# print(x.shape,y.shape)

batch_size = 64

# #download mnist data and split into train and test sets
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# all_digits = np.concatenate([x_train, x_test])
# all_labels = np.concatenate([y_train, y_test])

# # Scale the pixel values to [0, 1] range, add a channel dimension to
# # the images, and one-hot encode the labels.
# all_digits = all_digits.astype("float32") / 255.0
# all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
# all_labels = keras.utils.to_categorical(all_labels, 10)

# # Create tf.data.Dataset.
# dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
# dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# print(f"Shape of training images: {all_digits.shape}")
# print(f"Shape of training labels: {all_labels.shape}")