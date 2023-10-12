#required libraries; run command on cmd
#pip install numpy pandas matplotlib scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import preprocessing, svm
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

#mounting drive, used when in google colab
#from google.colab import drive
#drive.mount('/drive')

import csv

# Opeing csv and storing data
f = open('weatherdata.csv', 'r') #change the path
csvreader = csv.reader(f)
headings = [] 
rows = [] 
headings=next(csvreader)
for r in csvreader:
    rows.append(r)

#trial to see if loaded data is correct
pd.DataFrame(rows,columns=headings)[0:9] 


#regression work

x= np.array([])
for i in range(0,9):
  x = np.append(x,float(rows[i][4]))
x= x.reshape(-1,1)

y=np.array([])
for i in range(0,9):
  y=np.append(y,float(rows[i][1]))
#y=y.reshape(-1,1) #2d inputs, give out 2d slope lol

model = LinearRegression()
model.fit(x,y) #return self, i.e saved as model variable itself
"""r_sq = model.score(x,y)
print(f"coeff of determination: {r_sq}")
print(f"intercept c= {model.intercept_}")
print(f"slope m= {model.coef_}")"""

a=float(input("Enter area in sqft: "))
print(f"Price prediction in $: {model.predict(np.array([a]).reshape(-1,1))}")
