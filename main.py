print('hello world')


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/home/user/dl_madhurjya/deeplearning_campusx/datasets/placement.csv')

print(df.shape)
df.head()

plt.figure(figsize=(10,6))
sns.scatterplot(x='cgpa',y='resume_score',data=df, hue='placed')
plt.savefig("images/my_first_plot.png", dpi=300, bbox_inches="tight")