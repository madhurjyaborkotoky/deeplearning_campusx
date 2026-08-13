import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/home/user/dl_madhurjya/deeplearning_campusx/datasets/Admission_Predict_Ver1.1.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.duplicated().sum())

df.drop(columns=['Serial No.'], inplace=True)
print(df.head())

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(7,activation='relu',input_dim=7))
model.add(Dense(7,activation='relu'))
model.add(Dense(1,activation='linear'))

print(model.summary())

model.compile(loss='mean_squared_error',optimizer='Adam')

history = model.fit(X_train_scaled,y_train,epochs=100,validation_split=0.2)

y_pred = model.predict(X_test_scaled)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.savefig('/home/user/dl_madhurjya/deeplearning_campusx/admission_predict/images/model_loss.png')