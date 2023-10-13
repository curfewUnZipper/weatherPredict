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
f = open('weatherData.csv', 'r') #change the path
csvreader = csv.reader(f)
headings = [] 
rows = [] 
headings=next(csvreader)
for r in csvreader:
    rows.append(r)

#trial to see if loaded data is correct
pd.DataFrame(rows,columns=headings)[0:9] 



#multiple regression
#x-axis date-time in string
from datetime import datetime
x= np.array([])
for i in range(len(rows)):
  j = rows[i][0]
  try:
    j=datetime.strptime(j,"%m/%d/%Y %H:%M")
  #print(f"{j.day:02d}{j.month:02d}{j.year:04d}{j.hour:02d}") #example: 1805200314
  except ValueError:
    j=datetime.strptime(j,"%m/%d/%y %H:%M")
  j= int(f"{j.day:02d}{j.month:02d}{j.hour:02d}")
  #print(type(j),j)
  x = np.append(x,j)
x= x.astype(np.float64).reshape(-1,1)
#print(x)
#y axis array multi-dim of everything else
y=np.array([])
for i in range(len(rows)):
  y= np.append(y,rows[i][1:len(rows[i])-1])
y=y.astype(np.float64).reshape(-1,len(rows[0])-2)
#print(pd.DataFrame(y,columns=headings[1:len(headings)-1])[0:9])

model = LinearRegression().fit(x,y) #return self, i.e saved as model variable itself
#r_sq = model.score(x,y)
#print(f"index of determination: {r_sq}")
#print(f"intercept c= {model.intercept_}")
#print(f"slope m= {model.coef_}")

a=input("Enter Date-Time as 13/01/2023 13:00 -> ")
a=datetime.strptime(a,"%d/%m/%Y %H:%M")
a= int(f"{a.day:02d}{a.month:02d}{a.hour:02d}")
print(f"Weather Prediction: {pd.DataFrame(model.predict(np.array(a).reshape(-1,1)),columns=headings[1:-1])}")
