#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
keras.__version__


# In[2]:


import glob
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

neutral_dir = glob.glob("./emotion/neutral/*")
tired_dir = glob.glob("./emotion/tired/*")
happy_dir = glob.glob("./emotion/happy/*")
sad_dir = glob.glob("./emotion/sad/*")


# In[3]:


print(sad_dir)


# In[4]:


from keras.preprocessing import image
import numpy as np


# In[5]:


xsize=150
ysize=150
x=[]
y=[]

def imgtoarr(imgdir, label):
    for i in imgdir:
        img = image.load_img(i, target_size=(xsize,ysize))
        img_tr= image.img_to_array(img)        
        img_tr /= 255.
            
        y.append(label)
        x.append(img_tr)


# In[6]:


imgtoarr(neutral_dir, 0)
imgtoarr(tired_dir, 1)
imgtoarr(happy_dir, 2)
imgtoarr(sad_dir, 3)


# In[7]:


x=np.array(x)
y=np.array(y)


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

input_shape = (xsize, ysize, 3)

batch_size = 50
num_classes = 4
epochs = 15


# In[9]:


from keras.utils import to_categorical


# In[10]:


y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# In[11]:


from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten

conv_base = Sequential()

conv_base.add( VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3)) )


conv_base.add(Flatten())
conv_base.add(Dense(num_classes, activation='softmax'))
conv_base.summary()


# In[12]:


conv_base.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
conv_base.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))


# In[ ]:


score = conv_base.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:




