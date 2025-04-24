# SGD Regressor for Multivariate Linear Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the California housing dataset.Create a DataFrame, select features and multi-targets (AveOccup, target).Split into training and testing sets.
2. Standardize both input (X) and output (Y) data using StandardScaler.
3. Use SGDRegressor wrapped in MultiOutputRegressor to fit the training data.
4. Predict on test data, inverse transform predictions, and calculate Mean Squared Error (MSE).

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Prajin S
RegisterNumber: 212223230151
*/
```
```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScalerdata=fetch_california_housing()
print(data)
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
df.tail()
df.info()
x=df.drop(columns=['AveOccup','target'])
y=df[['AveOccup','target']]
x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=11)
x_train.shape
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.fit_transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.fit_transform(y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
Y_pred=multi_output_sgd.predict(x_test)
Y_pred
y_pred=scaler_y.inverse_transform(Y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error :",mse)
y_pred
```

## Output:

![image](https://github.com/user-attachments/assets/0076fc90-0da2-4cdc-8679-6024343a7d1b)


![image](https://github.com/user-attachments/assets/1d780096-0625-4e32-a293-ee8e5c577a76)


![image](https://github.com/user-attachments/assets/78092c17-5c82-4ec5-affa-d0dc4326711b)


![image](https://github.com/user-attachments/assets/0532b3c6-7615-4874-8baf-dc2423162fdf)

![image](https://github.com/user-attachments/assets/4bb31974-1ff4-4d44-bc88-561c4cac801c)

![image](https://github.com/user-attachments/assets/485be235-2f5e-4320-b29a-24d58acb6baf)

![image](https://github.com/user-attachments/assets/36ea94fb-a80d-4f14-8ce0-654971594b60)

![image](https://github.com/user-attachments/assets/7d4f978a-6079-4c3d-9d9d-ad5a5d359ea3)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
