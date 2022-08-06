from ArtGAN import app
from flask import render_template, request
import flask
import ArtGAN.process

from keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import cv2
# creates an super res object
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "C:/Users/Danie/Documents/ArtGAN/web/ArtGAN/FSRCNN_x4.pb"

# read and creates the model
sr.readModel(path)
sr.setModel("fsrcnn", 4)

model = load_model('C:/Users/Danie/Documents/ArtGAN/web/ArtGAN/cgan_generator200.h5',compile=False)

@app.route('/')

@app.route('/home')
def home():
    return render_template('home.html', title='ArtGAN')


@app.route('/generate')
def generate():
  latent_dim = 100
  [inputs,labels] = ArtGAN.process.generate_latent_points(latent_dim,100)
  labels = np.asarray([x for _ in range(10) for x in range(10)])
  X = model.predict([inputs,labels])
  X = (X + 1) / 2.0
  X = (X*255).astype(np.uint8)
  Y = []
  for item in X:
    result = sr.upsample(item)
    Y.append(result)
  Y = np.asarray(Y)
  for i in range(10*10):
    plt.axis('off')
    plt.subplot(10,10,i+1)
    plt.imshow(Y[i,:,:,:])
  plt.savefig('C:/Users/Danie/Documents/ArtGAN/web/ArtGAN/static/img.png')
  plt.close('all')
  print("DONE")
  return ("nothing")


@app.route('/about')
def about():
    return render_template('about.html', title='ArtGAN')

