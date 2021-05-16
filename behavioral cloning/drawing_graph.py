import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import  matplotlib
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_core.python.keras.callbacks import CSVLogger, Callback
import os
os.chdir('./dataset/raw')
# %matplotlib inline
# plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
x=np.load('y1.npy')
print(x)
# plt.hist(x, bins=50)
# plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
n, bins, patches = plt.hist(x, bins=6, edgecolor = 'black',color='#0504aa',)
plt.xlabel('steering values')
plt.ylabel('Frequency')
plt.title('steering values vs frequency')
# Set a clean upper y-axis limit.
plt.show()
