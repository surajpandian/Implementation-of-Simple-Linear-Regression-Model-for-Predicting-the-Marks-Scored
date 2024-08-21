# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries
2. Load and Explore the Data
3. Segregate Features and Target Variable
4. Split Data into Training and Testing Sets
5. Train a Linear Regression Model
6. Make Predictions
7. Compare Predicted and Actual Values
8. Plot the Training and Testing Data
9. Evaluate the Model
10. Output the Errors


## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: Suraj pandian R

RegisterNumber: 212223080040
*/

```python 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#read csv file
df= pd.read_csv('data.csv')
#displaying the content in datafile
df.head()
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred

#displaying actual values
y_test

#graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#graph plot for test data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print('MSE = ',mse)
print('MAE = ',mae)
print('RMSE = ',rmse)
```

## Output:
![alt text](1.png)


![alt text](2.png)


![alt text](3.png)


![alt text](4.png)


![alt text](5.png)


![alt text](6.png)


![alt text](9.png)


![alt text](<download (8).png>)


![alt text](<download (7)-1.png>)

# Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
