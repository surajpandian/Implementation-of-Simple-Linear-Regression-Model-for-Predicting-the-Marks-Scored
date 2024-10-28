# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
### Date:
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the program.

Step 2: Import the standard Libraries. 

Step 3: Set variables for assigning dataset values. 

Step 4: Import linear regression from sklearn.

Step 5: Assign the points for representing in the graph.

Step 6: Predict the regression for marks by using the representation of the graph.

Step 7: Compare the graphs and hence we obtained the linear regression for the given datas.

Step 8: Stop the program.


## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: R suraj pandian

RegisterNumber: 212223080040

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores.csv")
df.head()

df.tail()

# segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

# splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

# displaying predicted values
Y_pred

Y_test

# graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# graph plot for test data
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
### Head:
![201](https://github.com/user-attachments/assets/e4c4b618-7c2e-4fc9-b012-64641e7747ea)

### Tail:
![202](https://github.com/user-attachments/assets/eeb3d722-1049-42b3-86ab-38010ad9f9ae)

![203](https://github.com/user-attachments/assets/23e159a7-35de-4270-ad0e-f6e5095f8fcb)

### Array value of X:
![204](https://github.com/user-attachments/assets/1ab99bd8-b7a3-488f-9248-8388a05949f1)

### Array value of Y:
![205](https://github.com/user-attachments/assets/0bfabb0e-04bd-4582-9457-9941fee5bca7)

### Y prediction:
![206](https://github.com/user-attachments/assets/4b5a079a-f794-4892-9fe5-891a33e3b6d2)

### Training set graph:
![207](https://github.com/user-attachments/assets/ba2ba83c-549e-43bf-a35b-a9bc0515c2cc)

### Testing set graph:
![208](https://github.com/user-attachments/assets/d43852d6-0a05-4947-8ad7-f776470e9bd5)

### Values of MSE, MAE and RMSE:

![209](https://github.com/user-attachments/assets/2e58769c-9801-4de9-b8cd-312b7772de62)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
