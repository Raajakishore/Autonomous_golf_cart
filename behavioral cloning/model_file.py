import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_core.python.keras.callbacks import CSVLogger, Callback
import os
...
# # create a data generator
# datagen = ImageDataGenerator()
# # load and iterate training dataset
# train_it = datagen.flow_from_directory('./fotos',batch_size=64,classes=['train'])
# # load and iterate validation dataset
# val_it = datagen.flow_from_directory('./fotos', batch_size=64,classes=['validation'])
# # load and iterate test dataset
# test_it = datagen.flow_from_directory('./fotos',batch_size=64,classes=['test'])


# model = tf.keras.Sequential(
#     [        layers.BatchNormalization(),
#         layers.Cropping2D(((70,0),(0,0))),
#      layers.Conv2D(24,5,strides=2,activation='relu'),
#     layers.Conv2D(36,5,strides=2,activation='relu'),
#     layers.Conv2D(48,5,strides=2,activation='relu'),
#         layers.Conv2D(64, 3, activation='relu'),
#         layers.Conv2D(64, 3, activation='relu'),
#     layers.Flatten(),
# layers.Dense(100),
# layers.Dense(50,activation='relu'),
# layers.Dense(10,activation='relu'),
# layers.Dense(1,activation='sigmoid')     ]
# )
# # # model.summary()
model = keras.models.load_model('./dataset/modele2.h5')
# # with open('./steering angle.csv','r') as csvfile:
# #    lines = csv.reader(csvfile)
# #    y_train = np.zeros((2028,1),dtype='float32')
# #    y_test=  np.zeros((663,1),dtype='float32')
# #    i=0
# #    for l in lines:
# #      if i>0 and i<2049:
# #        y_train[i-1,0] = np.array(l[-1]).astype('float32')
# #      if i>2028 and i<2753:
# #          y_test[i-1,0] = np.array(l[-1]).astype('float32')
# #      i=1+1
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     0.1,
#     decay_steps=1000,
#     decay_rate=0.96,
#     staircase=True)
# model.compile(
#     loss='mse',optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),metrics=["accuracy"]
# )
x_train=(np.load('./dataset/raw/x_fipped.npy')).astype('float32')
y_train=(np.load('./dataset/raw/y_fipped.npy')/65535).astype('float32')
# print(layers.BatchNormalization()(y_train))
# print(layers.BatchNormalization()(y_train).shape)
# x_train,y_train=shuffle(x_train,y_train)
# print(x_train.shape)
# print(y_train.shape)
# csv_logger = CSVLogger('training_losses.csv',append=True)
# class prediction_history(Callback):
#     def __init__(self):
#         self.predhis = []
#     def on_epoch_end(self, epoch, logs={}):
#         self.predhis.append(model.predict(predictor_train))
# predictions= prediction_history()
model.fit(x_train,y_train,epochs=20,verbose=1)
# x_test = np.load('./fotos/test.npy')
#pre = model.predict(x_train[23000:],64,1,callbacks=[csv_logger])
#print(pre)
# model.save('modele2.h5')
model.save('modele3.h5')
