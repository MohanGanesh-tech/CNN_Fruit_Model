#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import numpy as np
import cv2
import random
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize
from tensorflow.keras.models import load_model
import time 


# In[ ]:


tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)


# In[ ]:


BASE_DIR=os.getcwd()
dataset_dir=os.path.join(BASE_DIR,"Dataset")
categories=os.listdir(dataset_dir)
print(dataset_dir,"\n",categories)


# In[ ]:


training_data = []
IMG_SIZE = 100
def create_training_data():
    print("Please Wait it will take some time")
    for category in categories:
        path = os.path.join(dataset_dir,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
                
            except Exception as e:
                print(e)
                pass
            
create_training_data()
print("\n \n Total Traning Data Length",len(training_data))


# In[ ]:


random.shuffle(training_data)


# In[ ]:


X = []
y = []

for features,labels in training_data:
    X.append(features)
    y.append(labels)
    
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y = np.array(y)


# In[ ]:


pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

del X,y


# In[ ]:


X = []
y = []

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

del labels,features,y_train,y_test,X,y

x_train = x_train / 255
x_test = x_test / 255


# In[ ]:


checkpoint_path="saved_models/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

def createmodel():
    model = Sequential()

    model.add(Conv2D(64, (3,3), input_shape = x_train.shape[1:],activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,3,3, activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(512,activation= 'relu'))

    model.add(Dense(512,activation= 'relu'))
    model.add(Dense(len(categories), activation='softmax'))

    model.compile(loss="mean_squared_error",optimizer='adam',metrics=['accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1, period=5)


    hist = model.fit(x_train, y_train_one_hot,batch_size=50, epochs=25, validation_split=0.3, callbacks=[cp_callback] )

    model.summary()

    model.evaluate(x_test, y_test_one_hot)[1]

    history_dict = hist.history
    print(history_dict.keys())
    
    return model

begin = time.time() 
model = createmodel()
time.sleep(1) 
end = time.time() 
print(f"Total runtime of the program is {end - begin}") 


# In[ ]:


latest = tf.train.latest_checkpoint(checkpoint_dir)

model.load_weights(latest)
model.summary()

model.evaluate(x_test, y_test_one_hot)[1]


# In[ ]:


image_add='C:/Users/Ganesh PC/Desktop/Final/Test/75_100.jpg'
my_image = plt.imread(image_add)
plt.imshow(my_image)
my_image_resized = resize(my_image, (IMG_SIZE,IMG_SIZE,1))
probabilities = model.predict(np.array( [my_image_resized,] ))

index = np.argsort(probabilities[0,:])
print("Predicted Class:", categories[index[len(categories)-1]],"\nProbability:", probabilities[0,index[len(categories)-1]]*100)


# In[ ]:


import winsound
import time
while(True):
    frequency = 2500
    duration = 200
    winsound.Beep(frequency, duration)
    time.sleep(0.6)


# In[ ]:





# In[ ]:




