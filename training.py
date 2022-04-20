import numpy as np

data=np.load('data.npy')
target=np.load('target.npy')

import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten, Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#kernel_regularizer = keras.regularizers.l1()

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#kernel_regularizer = keras.regularizers.l2(0.01)

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(50,activation='relu'))

model.add(Dense(2,activation='softmax'))
#kernel_regularizer = keras.regularizers.l1_l2(0.1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


#from sklearn.model_selection import train_test_split
#train_data,test_data,train_target,test_target= train_test_split(data,target,test_size=0.1)
from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)

checkpoint = ModelCheckpoint('model-{epoch:03d}.model' , monitor = 'val_loss' , verbose = 0, save_best_only=True, mode='auto')
history=model.fit(train_data,train_target,batch_size=1,epochs=15 , callbacks=[checkpoint] , validation_split=0.2)



print(model.evaluate(test_data,test_target))
