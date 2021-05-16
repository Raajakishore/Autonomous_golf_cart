import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
# # plt.imshow(np.fliplr((np.load('./fipped_x.npy')[100]*65535).astype('uint8')))
# # plt.show()
# # a=y<12000
# # y_train=a*y
# # ind=np.nonzero(y_train)[0]
# # y=y_train[ind]
# # x=x[ind]
# # y1=65535-(y/65535)
# # x=np.fliplr(x[:])
# # np.save('fipped_x.npy',x)
# # np.save('fipped_y.npy',y1)
# # x=(np.load('./fipped_x.npy')*65535).astype('uint8')
os.chdir('./dataset/raw')
#
# # np.save('sample_use',x[:10])
# # plt.imshow()
# # plt.show() 
# # x=np.load('x5000.npy')
# # x1=np.load('x10000.npy')
# # x2=np.load('x15000.npy')
#
# # x3=np.load('x20000.npy')
# # x = (np.load('x1.npy')*65535).astype('uint8')
# #
# # print(x[100])
# # print(x.dtype)
#
x_final=np.append(np.load('x1.npy'),np.load('x_fipped.npy'),0)
y_final=np.append(np.load('y1.npy'),np.load('y_fipped.npy'),0)
print(x_final.shape)
print(y_final.shape)
x_final,y_final=shuffle(x_final,y_final)
np.save('x_final1.npy',x_final[:16500])
np.save('y_final1.npy',y_final[:16500])
np.save('x_final2.npy',x_final[16500:])
np.save('y_final2.npy',y_final[16500:])


# np.save('D:/x_final.npy',x_final)
#
#
# # plt.imshow(np.fliplr(np.flipud(x[2])))
# # print(x[2])
# # plt.show()
# y_train=np.load('./dataset/raw/y_fipped.npy')
# y_train= 65535-((65535-y_train)*65535)
# np.save('y_fipped.npy',y_train)
print(y_train)