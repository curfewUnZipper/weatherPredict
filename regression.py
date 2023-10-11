

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression




df = pd.read_csv('data.csv')
df_binary = df[['price', 'bedrooms']]
 
# Taking only the selected two attributes from the dataset
df_binary.columns = ['Price', 'Bedrooms']
#display the first 5 rows
df_binary.head(5)




#plotting the Scatter plot to check relationship between Sal and Temp
sns.lmplot(x ="Bedrooms", y ="Price", data = df_binary, order = 2, ci = None)
plt.show()
df_binary.fillna(method ='ffill', inplace = True)




X = np.array(df_binary['Bedrooms']).reshape(-1, 1)
y = np.array(df_binary['Price']).reshape(-1, 1)

# Separating the data into independent and dependent variables
# Converting each dataframe into a numpy array 
# since each dataframe contains only one column
df_binary.dropna(inplace = True)

# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Splitting the data into training and testing data
regr = LinearRegression()

regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))





y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()
# Data scatter of predicted values










df_binary500 = df_binary[:][:500]

# Selecting the 1st 500 rows of the data
sns.lmplot(x ="Price", y ="Bedrooms", data = df_binary500, order = 2, ci = None)


df_binary500.fillna(method ='fill', inplace = True)

X = np.array(df_binary500['Price']).reshape(-1, 1)
y = np.array(df_binary500['Bedrooms']).reshape(-1, 1)

df_binary500.dropna(inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))



y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()





from sklearn.metrics import mean_absolute_error,mean_squared_error

mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
#squared True returns MSE value, False returns RMSE value.
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)


