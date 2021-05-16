import cv2
import numpy as np
import os
os.chdir('C:/Users/raaja/Anaconda3/envs/IntroToTensorFlow/fotos/train')
images = os.listdir('.')
a=cv2.imread(images[0])
a=np.expand_dims(a,0)
a=a.astype('float32')
print(a.shape)
for i in range(1,len(images)):
    b=cv2.imread(images[i])
    b=np.expand_dims(b,0)
    b=b.astype('float32')
    a=np.concatenate((a,b))
    print(i)
np.save('C:/Users/raaja/Anaconda3/envs/IntroToTensorFlow/fotos/train.npy',a)


