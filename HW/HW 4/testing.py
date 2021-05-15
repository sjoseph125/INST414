import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import xgboost

df_train = pd.read_csv('train.csv', index_col=0)
df_test = pd.read_csv('test.csv', index_col=0)

correlation=df_train.corr()['SalePrice']
top_corr=correlation[correlation.values>0.5].sort_values(ascending=False)

df_test['GarageCars']=df_test['GarageCars'].fillna(df_test['GarageCars'].mode()[0])
df_test['TotalBsmtSF']=df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mean())
df_test['GarageArea']=df_test['GarageArea'].fillna(df_test['GarageArea'].mean())

combined_df=pd.concat([df_train, df_test])

combined_dummies=pd.get_dummies(combined_df[top_corr.index])
combined_dummies.columns

df_train=combined_dummies.iloc[:1459]
df_test=combined_dummies.iloc[1460:]
df_test=df_test.drop(['SalePrice'], axis=1)

x_train=df_train.drop(['SalePrice'], axis=1)
y_train=df_train['SalePrice']


LinReg = linear_model.LinearRegression()

LinReg.fit(x_train, y_train)
y_pred=LinReg.predict(df_test)

# print(mean_squared_error(y_train, y_pred))

classifier=xgboost.XGBRegressor()
classifier.fit(x_train, y_train)
y_pred2 = classifier.predict(df_test)
print(mean_squared_error(y_train, y_pred2))

# submission = pd.read_csv('sample_submission.csv')
# submission.drop(['SalePrice'], axis=1)
# submission['Id']=df_test.index
# submission['SalePrice']=y_pred2
# submission.to_csv('SamsonJ_INST414.csv', index = False)
#print(submission)
sns.scatterplot(x=df_test.index, y=y_pred)
plt.show()