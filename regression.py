#required libraries; run command on cmd
#pip install numpy pandas scikit-learn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
from datetime import datetime

rows= pd.read_csv("https://raw.githubusercontent.com/curfewUnZipper/weatherPredict/0df5cc7cb69d74e1b820fc6d8eec555461192dea/weatherData.csv")
cols = list(rows.columns)


#multiple regression
#x-axis date-time in string
from datetime import datetime

x = rows[cols[0]]
x = np.array(x)
#print(x)
for i in range(len(x)):
  #print(x[i])
  try:
    x[i]=datetime.strptime(x[i],"%m/%d/%Y %H:%M")
  #print(f"{j.day:02d}{j.month:02d}{j.year:04d}{j.hour:02d}") #example: 1805200314
  except ValueError:
    x[i]=datetime.strptime(x[i],"%m/%d/%y %H:%M")
  x[i]= int(f"{x[i].day:02d}{x[i].month:02d}{x[i].hour:02d}")
  #print(type(x[i]),x[i])
x= x.astype(np.float64).reshape(-1,1)
#print(len(x))




#y axis array multi-dim of everything else
y=np.array([])
#print(rows['Temp_C'][0])
for i in range(len(x)): #iterate through data
  k = [] #to store a row
  for j in range(1,len(cols)-1): #move in a single row
    k.append(rows[cols[j]][i])
  y= np.append(y,k)
y=y.astype(np.float64).reshape(-1,len(cols)-2)
#print(pd.DataFrame(y,columns=cols[1:len(cols)-1])[0:9])

#training model
model = LinearRegression().fit(x,y) #return self, i.e saved as model variable itself
r_sq = model.score(x,y)
print(f"index of determination: {r_sq}")
#print(f"intercept c= {model.intercept_}")
#print(f"slope m= {model.coef_}")

a=input("Enter Date-Time (For Example: 13/01/2023 13:00) -> ")
a=datetime.strptime(a,"%d/%m/%Y %H:%M")
a= int(f"{a.day:02d}{a.month:02d}{a.hour:02d}")
print(f"Weather Prediction:\n{pd.DataFrame(model.predict(np.array(a).reshape(-1,1)),columns=cols[1:-1])}")
print("This is accurate prediction of weather in banglore")
