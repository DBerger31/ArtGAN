from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Embedding, Concatenate, BatchNormalization
from matplotlib import pyplot as plt

from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np

import time

start_time = time.time()

# Loads our custom dataset of paintings and their respective labels
def load_custom_data():
    x = np.load('images.npy')
    y = np.loadtxt('labels.txt')
    return x, y

# trainX, trainy = load_custom_data() # shape of (62145, 64, 64, 3) and (62145,)

# plot the first 25 images of our dataset 
# for i in range(25):
# 	plt.subplot(5, 5, 1 + i)
# 	plt.axis('off')
# 	plt.text(0.5,0.5,trainy[i],horizontalalignment='center', verticalalignment='center')
# 	plt.imshow(trainX[i])
# plt.show()


''' 
Creates the discriminator model using keras functional API 
Arg1: Shape of the discriminator input
Arg2: Number of classes the dataset has
'''
def define_discriminator(in_shape=(64,64,3), n_classes=5):
	
    in_label = Input(shape=(1,)) # the label input

    # embedding layer added to create a vector of size 50 for each label we have (random values that acts as another set of weights)
    li = Embedding(n_classes, 50)(in_label) 

    li = Dense(in_shape[0] * in_shape[1])(li) # 64 x 64 = 4096

    li = Reshape((in_shape[0], in_shape[1], 1))(li) # converts our dense layer to a shape of 64x64x1

    in_image = Input(shape=in_shape) # our image input is 64x64x1

    merge = Concatenate()([in_image, li]) # merges the 2 input layers -> 64 x 64 x 4

    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge) 
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    fe = Conv2D(128,(3,3),strides=(2,2),padding='same')(fe) #increased the number of filters from 128 -> 258
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    fe = Conv2D(128,(3,3),strides=(2,2),padding='same')(fe) #increased the number of filters from 128 -> 258
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    # flattens multidimensional input into a single dimension
    fe = Flatten()(fe) 
    fe = Dropout(0.2)(fe)


    # output
    out_layer = Dense(1, activation='sigmoid')(fe) 

    # define model
    # image and and label as inputs and output for it
    model = Model([in_image, in_label], out_layer)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# test_discr = define_discriminator()
# test_discr.summary()

def define_generator(latent_dim, n_classes=5):
    
    in_label = Input(shape=(1,))  # the label input

    # embedding layer added to create a vector of size 50 for each label we have (random values that acts as another set of weights)
    li = Embedding(n_classes, 50)(in_label) 

    n_nodes = 8 * 8  # we need this number to be a factor of the image dimensions
    li = Dense(n_nodes)(li)
    li = Reshape((8, 8, 1))(li)


    # latent vector input with dimension of 100, which is standard
    in_lat = Input(shape=(latent_dim,)) 


    n_nodes = 128 * 8 * 8
    gen = Dense(n_nodes)(in_lat) 
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((8, 8, 128))(gen) 
    # merge image gen and label input
    merge = Concatenate()([gen, li]) 

    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge) #32x32x128
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dropout(0.2)(gen)

    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen) #64x64x128 , decreased filter from 128 -> 64
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dropout(0.2)(gen)

    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen) #64x64x128 , decreased filter from 128 -> 64
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dropout(0.2)(gen)

    # output
    out_layer = Conv2D(3, (8,8), activation='tanh', padding='same')(gen) #64x64x3
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model   # Model is not compiled becuase it is only trained within the GAN

# test_gen = define_generator(100, n_classes=10)

def define_gan(g_model, d_model):
	d_model.trainable = False  # discriminator is trained seperately
	gen_noise, gen_label = g_model.input  # latent vector and amount of labels from generator
	
	gen_output = g_model.output  # image output from generator
    
	# generator output and generator label is the input to the discriminator model
	gan_output = d_model([gen_output, gen_label])
	# Gan takes noise and label is input and outputs a class
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def load_real_samples():
  trainX, trainy = load_custom_data()
  X = trainX.astype('float32')
  # scale from [0,255] to [-1,1]
  X = (X - 127.5) / 127.5
  return [X, trainy]


'''
Randomly picks some of the real images to train the GAN
real images are assigned a label of 1
Arg1: the data set
Arg2: the number of real samples we wanna generate
Output: the images as X, the label as labels and y is a 1 because its a real image
'''
def generate_real_samples(dataset, n_samples):
	images, labels = dataset
	ix = randint(0, images.shape[0], n_samples)
	X, labels = images[ix], labels[ix]
	y = ones((n_samples, 1))
	return [X, labels], y

''' 
Creates noise for the generator to create images from
Arg1: latent dimension size (usually 100)
Arg2: the number of samples
Arg3: the amount of classes
Output: the noise and the labels
'''
def generate_latent_points(latent_dim, n_samples, n_classes=5):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

'''
Creates fake images to train the GAN
Fake images are assigned a label of 0
Arg1: the data set
Arg2: the number of fake samples we wanna generate
Output: the images as X, the label as labels and y is a 0 because its a fake image
'''
def generate_fake_samples(generator, latent_dim, n_samples):
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	images = generator.predict([z_input, labels_input])
	y = zeros((n_samples, 1))  #
	return [images, labels_input], y


d_losses = [] # for plotting the losses
g_losses = []

'''
Trains our discriminator and our generator by looping through a specified number of epochs and batch size
trains the discriminator on real and fake images each in a different batch
we generate a set of images using the generator and feed that back into the discriminator along with some real images we train the discriminator with before
We get our loss for the discriminator and generator
Args: gen model, disc model, gan model, dataset, latent dimension, epochs, and batch size
Output: No output but we save our generator model for future use
'''
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
      d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
      d_losses.append(d_loss)
      g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
      g_losses.append(g_loss)
      print('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f' %
        (i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
  g_model.save('10_mod.h5')

# Parameters
latent_dim = 100
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
dataset = load_real_samples()

# train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200)

# Load the trained model and generate a few images
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np
import matplotlib as pyplot

def save_plot(X,n):
  for i in range(n*n):
    plt.axis('off')
    plt.subplot(n,n,i+1)
    plt.imshow(X[i,:,:,:])
  plt.show()

def plot_losses():
  plt.figure(figsize=(10, 5))
  plt.title("Generator and Discriminator Loss During Training")
  plt.plot(d_losses, label="D")
  plt.plot(g_losses, label="G")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()


[inputs,labels] = generate_latent_points(latent_dim,100)
labels = np.asarray([x for _ in range(10) for x in range(10)])
model = load_model('200_mod.h5')
X = model.predict([inputs,labels])
X = (X + 1) / 2.0
X = (X*255).astype(np.uint8)

elapsed = (time.time() - start_time)
time = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
print("Process finished --- %s --- " % (time))

save_plot(X,10)
plot_losses()


