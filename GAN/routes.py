import os
from GAN import app
from flask import render_template, request
import flask
from GAN import process

from keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import cv2
# creates an super res object
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = os.path.join(os.path.dirname(__file__), '..', 'GAN', 'FSRCNN_x4.pb')

# read and creates the model
sr.readModel(path)
sr.setModel("fsrcnn", 4)

model = load_model(os.path.join(os.path.dirname(__file__), '..', 'GAN', 'cgan_generator200.h5'),compile=False)

@app.route('/')

@app.route('/home')
def home():
    return render_template('home.html', title='ArtGAN')

@app.route('/generate')
def generate():
  latent_dim = 100
  [inputs,labels] = process.generate_latent_points(latent_dim,10)
  labels = np.asarray([x for x in range(10)])
  X = model.predict([inputs,labels])
  X = (X + 1) / 2.0
  X = (X*255).astype(np.uint8)
  Y = []
  for item in X:
    result = sr.upsample(item)
    Y.append(result)
  Y = np.asarray(Y)
  for i in range(10):
    plt.axis('off')
    # plt.subplot(1,10,i+1)
    plt.imshow(Y[i])
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'GAN/static', f'img{i}.png'))
  plt.close('all')
  print("DONE")
  return ("nothing")


@app.route('/about')
def about():
    return render_template('about.html', title='ArtGAN')

