"""This file contains the experimental MNIST numbers cGAN model"""

from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Embedding, Concatenate
from matplotlib import pyplot as plt

from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np

#load in data
(trainX, trainy), (testX, testy) = load_data()

# display images
for i in range(25):
    plt.subplot(5, 5, 1 + i)
    plt.axis('off')
    plt.imshow(trainX[i])
plt.show()

# create the discriminator
def define_discriminator(in_shape=(28, 28, 1), n_classes=10):

    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)

    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)

    li = Reshape((in_shape[0], in_shape[1], 1))(li)

    in_image = Input(shape=in_shape)

    merge = Concatenate()([in_image, li])

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)

    # downsample
    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)

    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)

    # output
    out_layer = Dense(1, activation='sigmoid')(fe)

    model = Model([in_image, in_label], out_layer)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model

test_discr = define_discriminator()

# create the generator
def define_generator(latent_dim, n_classes=10):
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = 7 * 7
    li = Dense(n_nodes)(li)
    li = Reshape((7, 7, 1))(li)

    in_lat = Input(shape=(latent_dim,))

    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)
    merge = Concatenate()([gen, li])
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2),
                          padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    out_layer = Conv2D(1, (7, 7), activation='tanh', padding='same')(gen)
    model = Model([in_lat, in_label], out_layer)
    return model

test_gen = define_generator(100, n_classes=10)

# create GAN
def define_gan(g_model, d_model):
    d_model.trainable = False

    gen_noise, gen_label = g_model.input  
    gen_output = g_model.output 

    gan_output = d_model([gen_output, gen_label])
    model = Model([gen_noise, gen_label], gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

#loads real dataset samples
def load_real_samples():
    (trainX, trainy), (_, _) = load_data()
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5
    return [X, trainy]

#collects images from training dataset
def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1)) 
    return [X, labels], y

# generates noise from of which the generator will produce images
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

# creates generated images to test on the discriminator
def generate_fake_samples(generator, latent_dim, n_samples):
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input])
    y = zeros((n_samples, 1))
    return [np.array(images), labels_input], y


#train the model
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2) 

	for i in range(n_epochs):
		for j in range(bat_per_epo):
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
			d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
			y_gan = ones((n_batch, 1))
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			print('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
	g_model.save('mod.h5')


latent_dim = 100
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
dataset = load_real_samples()
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1)


# Load the trained model and generate images
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np
import matplotlib as pyplot

# load model
model = load_model('mod.h5',compile=False)
# generate images
def save_plot(X,n):
  for i in range(n*n):
    plt.axis('off')
    plt.subplot(n,n,i+1)
    plt.imshow(X[i,:,:,0],cmap='gray_r')
  plt.show()

#generate images
latent_dim = 100
[inputs,labels] = generate_latent_points(latent_dim,100)
labels = np.asarray([x for _ in range(10) for x in range(10)])
model = load_model('mod.h5')
X = model.predict([inputs,labels])
save_plot(X,10)