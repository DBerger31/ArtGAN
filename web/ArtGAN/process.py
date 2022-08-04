from ArtGAN import app
from flask import render_template, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
import flask
from matplotlib import pyplot as plt
from numpy.random import randn
from numpy.random import randint


def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]



