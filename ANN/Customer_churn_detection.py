#customer churn detection using artificial neural network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/home/user/dl_madhurjya/deeplearning_campusx/datasets/Churn_Modelling.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.duplicated().sum())