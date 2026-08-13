#handwritten digit recognition using keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import tensorflow
import keras
from keras import Sequential
from keras.layers import Dense, Flatten


(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
print(x_train)
print(x_train.shape)
print(y_train)

import matplotlib.pyplot as plt
plt.imshow(x_train[2])
plt.savefig("/home/user/dl_madhurjya/deeplearning_campusx/ANN/images/sample_digit.png", dpi=300, bbox_inches="tight")

x_train = x_train/255
x_test = x_test/255

model = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='leaky_relu'))
model.add(Dense(128,activation='leaky_relu'))
model.add(Dense(10,activation='softmax'))

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adamw',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=2000,validation_split=0.2)

y_prob = model.predict(x_test)

y_pred = y_prob.argmax(axis=1)

print('accuracy of the model is:', accuracy_score(y_test,y_pred))

plt.figure(figsize=(15,7))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.legend()
plt.grid()
plt.savefig("/home/user/dl_madhurjya/deeplearning_campusx/ANN/images/digit_loss_plot.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(20,9))
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.legend()
plt.grid()
plt.savefig("/home/user/dl_madhurjya/deeplearning_campusx/ANN/images/digit_accuracy_plot.png", dpi=300, bbox_inches="tight")