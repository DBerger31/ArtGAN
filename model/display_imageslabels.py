import matplotlib.pyplot as plt 
import numpy as np 


# Loads our custom dataset of paintings and their respective labels
def load_custom_data():
    x = np.load('images.npy')
    y = np.loadtxt('labels.txt')
    return x, y

trainX, trainy = load_custom_data() # shape of (62145, 64, 64, 3) and (62145,)

label = trainy
label = ['abstract' if i == 0 
        else 'landscape' if i == 1
        else 'religious painting' if i == 2
        else 'portrait' if i == 3
        else 'sketch and study' if i == 4
        else i for i in label]

# plot the first 25 images of our dataset 
for i in range(25):
    plt.subplot(5, 5, 1 + i)
    plt.axis('off')
    plt.text(0.5, 0.5, trainy[i], horizontalalignment='center', verticalalignment='center')
    plt.text(28, 0, label[i])
    plt.imshow(trainX[i])
plt.show()