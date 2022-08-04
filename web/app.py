from ArtGAN import app
from flask import Flask, render_template, request
import os

from keras.models import load_model 
from keras.preprocessing import image
import numpy as np

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0')