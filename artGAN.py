import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, LeakyReLU, BatchNormalization, Input, Embedding, Reshape, Concatenate, Dropout, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from numpy import expand_dims
# Testing Dataset
from keras.datasets import mnist
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from numpy import expand_dims
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data

'''
Functional API implementation which specifices the layer(arg1)(arg2)

arg1: input
arg2: output
'''   
# define the standalone discriminator model
def create_discriminator(img_shape, n_labels):

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
    print(f"This is the shape of label_input : {label_input.shape}")
    # image input
    img_input = Input(shape=img_shape)
    print(f"This is the shape of img_input : {img_input.shape}")
    # combine our two inputs an additional channel with be for the labels
    merge = Concatenate()([img_input, label_input])
    print(f"This is the shape of merge : {merge.shape}")

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

def create_generator(latent_dim, n_labels):
    label = Input(shape=(1,))
    label_input = Embedding(n_labels, 50)(label)
    # needs to be a factor of 28 (the image dimsensions)
    n_nodes = 7 * 7
    label_input = Dense(n_nodes)(label_input)
    label_input = Reshape((7,7,1))(label_input)
    img_input = Input(shape = (latent_dim,))

    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(img_input)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7,7,128))(gen)

    merge = Concatenate()([gen, label_input])
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    output = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
    model = Model([img_input, label_input], output) # Arg1: input, Arg2: output

    return model

def create_gan(discriminator, generator):
    # Trained seperately from the Generator
    discriminator.trainable = False
    gen_noise, gen_label_input = generator.input
    gen_output = generator.output
    
    print("THIS IS GEN SHAPES")
    print(gen_noise.shape) #(None, 100)
    print(gen_label_input.shape) # (None, 7, 7, 1)
    print(gen_output.shape) # (None, 28, 28, 1)

    # DOES NOT LIKE THE CONFLICTING DIMENSIONS HERE FOR SOME REASON
    gan_output = discriminator([gen_output, gen_label_input])

    model = Model([gen_noise, gen_label_input], gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model

def load_mnist():
    (x_train, y_train), (_, _) = load_data()

    # for testing the data set
    plt.imshow(x_train[0])
    plt.show()


    print(x_train.shape)
    print(x_train[0].shape)
    print(y_train[0])

    x_train = expand_dims(x_train, axis=-1)
    x_train = x_train.astype('float32')
    x_train = (x_train - 127.5) / 127.5
    return [x_train, y_train]

def generate_real_samples(dataset, n_samples):
    # choose random instances
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    # select images
    x, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [x, labels], y

def generate_latent_points(latent_dim, n_samples, n_labels):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = randint(0, n_labels, n_samples)
    return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
    z_input, labels = generate_latent_points(latent_dim, n_samples)
    imgs = generator.predict([z_input, labels])
    y = zeros((n_samples, 1))
    return [imgs, labels], y

def train(generator, discriminator, gan, dataset, latent_dim, n_epochs, n_batch):
    batch_per = int(dataset.shape[0] / n_batch)
    batch_half = int(n_batch / 2)
    for epoch in range(n_epochs):
        for batch in range(batch_per):
            [x, y], real = generate_real_samples(dataset, batch_half)

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

# print(x.shape,y.shape)


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