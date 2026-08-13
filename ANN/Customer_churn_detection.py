#customer churn detection using artificial neural network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/home/user/dl_madhurjya/deeplearning_campusx/datasets/Churn_Modelling.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.duplicated().sum())

print(df['Exited'].value_counts())

print(df['Geography'].value_counts())

print(df['Gender'].value_counts())

df.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)
print(df.head())

df = pd.get_dummies(df,columns=['Geography', 'Gender'], drop_first=True)

print(df.head())

x = df.drop(columns=['Exited'])
y = df['Exited']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=1)

print(x)
print(y)

print(x_train.shape)
print(y_train.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled)


import tensorflow
import keras
from keras import Sequential
from keras.layers import Dense, Flatten

model = Sequential()

model.add(Dense(11,activation='leaky_relu',input_dim = 11))
model.add(Dense(11,activation='leaky_relu'))
model.add(Dense(11,activation='leaky_relu'))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())

model.compile(loss='mse',optimizer='AdamW',metrics=['accuracy'])

history = model.fit(x_train_scaled,y_train,epochs=200, validation_split=0.2)

print(model.layers[0].get_weights())

print(model.predict(x_test_scaled))

y_log = model.predict(x_test_scaled)

y_pred = np.where(y_log>0.5, 1, 0)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

print(history.history)

plt.figure(figsize=(15,9))
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.legend()
plt.grid()
plt.savefig("/home/user/dl_madhurjya/deeplearning_campusx/ANN/images/loss_plot.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(15,9))
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.legend()
plt.grid()
plt.savefig("/home/user/dl_madhurjya/deeplearning_campusx/ANN/images/accuracy_plot.png", dpi=300, bbox_inches="tight")