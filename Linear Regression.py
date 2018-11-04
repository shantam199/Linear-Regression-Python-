# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 09:31:24 2018

@author: Bp_shantam Malgaonka
"""

#LINEAR REGRESSION

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

df = pd.read_csv('C:/Users/Bp_shantam Malgaonka/Desktop/SHANTAM/UdUdemy Courses/[FreeTutorials.Us] python-for-data-science-and-machine-learning-bootcamp/15 Linear Regression/USA_Housing.csv')

df.head()     #first 4-5 rows of dataframe df

df.info()

df.describe() #gives you the statistical information about the columns like mean,mode,median

df.columns  #gives you the column names of the df dataframe

sns.pairplot(df)

sns.distplot(df['Price']) #gives the distribution of the price column

df.corr() #to get the correlation matrix

sns.heatmap(df.corr(),annot=True)

df.columns

X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y=df['Price']
#ok so here we have to take X and y to train our model we are not taking the address column as it is in text
#format and so we have to choose the target column which is price..why price because we want to predict the price
#of the houses so that is the main reason and other columns in to the X

#now we are going to run the train_test_split function  which will do the
#training of the model and then testing of the model

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

#now we have to train the model

from sklearn.linear_model import LinearRegression

lm= LinearRegression()

lm.fit(X_train, y_train)

#Out[34]: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False) then it means your linear regression model is trained

print(lm.intercept_)  #gives you the intercepts

lm.coef_                #gives you the Coefficients

X_train.columns

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


#PREDICTIONS

predictions = lm.predict(X_test)

predictions


plt.scatter(y_test,predictions)


sns.distplot(y_test-predictions)


#linear Regression Excersice project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

customers = pd.read_csv('C:/Users/Bp_shantam Malgaonka/Desktop/SHANTAM/UdUdemy Courses/[FreeTutorials.Us] python-for-data-science-and-machine-learning-bootcamp/15 Linear Regression/customers.csv')


customers.head()
customers.describe()
customers.info()

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)



sns.jointplot(x='Time on App',y='Length of Membership',data=customers,kind='hex')


sns.pairplot(customers)

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


#Training and Testing Data

customers.columns

y=customers['Yearly Amount Spent']

X=customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

#use train_test_split function to training and testing the model
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression


lm=LinearRegression()  #creating an instance of linear Regression Model

lm.fit(X_train,y_train)

lm.intercept_

lm.coef_


#Predicting from the Test data

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y-Test(True Values)')
plt.ylabel('Predictions(Predicted Values)')


#Evaluating the Model

from sklearn import metrics

print('MAE',metrics.mean_absolute_error(y_test,predictions))
print('MSE',metrics.mean_squared_error(y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,predictions)))

sns.distplot(y_test-predictions,bins=50)

#Concolusion
cdf1 = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])

cdf1

