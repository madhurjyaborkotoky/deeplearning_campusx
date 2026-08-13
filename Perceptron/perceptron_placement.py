
#Training a perceptron model on the placement dataset and plotting the decision regions of the model.



#importing libraries
from pprint import pp

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions


#reading the dataset
df = pd.read_csv('/home/user/dl_madhurjya/deeplearning_campusx/datasets/placement.csv')

#getting the shape and brief idea of the dataset
print(df.shape)
print(df.head())

#plotting the scatter plot of cgpa vs resume score
plt.figure(figsize=(10,6))
sns.scatterplot(x='cgpa',y='resume_score',data=df, hue='placed')
plt.savefig("/home/user/dl_madhurjya/deeplearning_campusx/Perceptron/images/dataset_scatter_plot.png", dpi=600, bbox_inches="tight")

#splitting the dataset into features and target variable
x = df.iloc[:,0:2]
y = df.iloc[:,-1]

#creating the perceptron model and fitting it to the dataset
p = Perceptron()
p.fit(x, y)

#calculating the weight and bias of the model
print(p.coef_)
print(p.intercept_)

#plotting the decision regions of the perceptron model
plot_decision_regions(x.values, y.values, clf=p)
plt.savefig("/home/user/dl_madhurjya/deeplearning_campusx/Perceptron/images/decision_regions.png", dpi=300, bbox_inches="tight")

