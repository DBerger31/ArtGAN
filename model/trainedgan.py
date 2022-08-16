from artgan import generate_latent_points

# Load the trained model and generate a few images
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np
import matplotlib as plt

model = load_model('mod.h5',compile=False)
# generate images
def save_plot(X,n):
  for i in range(n*n):
    plt.axis('off')
    plt.subplot(n,n,i+1)
    plt.imshow(X[i,:,:,0])
  plt.show()
latent_dim = 100
[inputs,labels] = generate_latent_points(latent_dim,100)
labels = np.asarray([x for _ in range(10) for x in range(10)])
model = load_model('mod.h5')
X = model.predict([inputs,labels])
save_plot(X,10)