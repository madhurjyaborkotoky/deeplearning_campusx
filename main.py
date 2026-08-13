print('hello world')

#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#reading the dataset
df = pd.read_csv('/home/user/dl_madhurjya/deeplearning_campusx/datasets/placement.csv')

#getting the shape and brief idea of the dataset
print(df.shape)
print(df.head())

#plotting the scatter plot of cgpa vs resume score
plt.figure(figsize=(10,6))
sns.scatterplot(x='cgpa',y='resume_score',data=df, hue='placed')
plt.savefig("images/dataset_scatter_plot.png", dpi=300, bbox_inches="tight")

