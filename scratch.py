import numpy as np
import pandas as pd
# import csv
# from scipy.stats import norm
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.stats import entropy
data = pd.read_csv('midterm_data.csv')

# print(data.groupby('Make')['Engine HP'].mean())
#print(data.isnull().sum())

df_1 = data.sample(n=1000, random_state=99)
original_sample = df_1.MSRP
means_new_samples = []
np.random.seed(11)
for i in range(1000):
             new_sample = np.random.choice(original_sample, 1000)
             means_new_samples.append(new_sample.mean())

conf_int = np.percentile(means_new_samples,[2.5,97.5])
print(conf_int)

p=[1/6,1/6,1/6,1/6,1/6,1/6]
#print(entropy(p, base=2))




# mean = [0,0]
# cov = [[1,0], [0,1]]
# x,y=np.random.multivariate_normal(mean, cov, 10000).T
# plt.hist2d(x,y, bins=30, cmap='Blues')
# plt.show()
# a3 = np.array([ [[2,4,8], [9, 3, 5], [7, 6, 8]], 
#                [[8, 1,9], [6, 7, 7], [5, 9, 8]], 
#                [[5,0,1],[3, 0, 3], [2, 3, 8]]])
# a4=[[8, 5, 8], [9, 7, 8], [1, 3, 8]]

