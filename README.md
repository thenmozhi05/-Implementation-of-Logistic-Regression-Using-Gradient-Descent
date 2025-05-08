# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: GAUTHAM KRISHNA S
RegisterNumber:  212223240036
*/
```
```PY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
Y

theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta

theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred

y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(y_pred)

print(Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

## Output:

### Read the file and display

![324679994-1f016359-1822-4e61-a4bd-2e42b29b2193](https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/8460f007-dd72-418a-a1a4-89e7c352c482)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>



### Categorizing columns
![324680196-803932f9-6766-466c-b982-4a5d4a7f8116](https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/c3b56cef-53f7-4542-b0cc-1bb02844df55)


### Labelling columns and displaying dataset
![324680578-3652a512-7da7-424c-9669-89e9451fbdcd](https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/e0c05953-83f6-49f7-96b9-95a4a128e9ea)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>


### Display dependent variable
![324680688-e7b177aa-1745-4be5-8193-8ffd8f529775](https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/2b65a530-46a7-4bdb-b0c5-6c2bec37fe19)

### Accuracy
![324680804-98088e79-96c4-4b0c-a6fe-9375a8f80849](https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/a05d9cb7-87e2-4d7b-a2fd-360516088c97)


### Printing Y
![324680946-4ad464f1-fd31-45c4-8b6e-200e80a56472](https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/d31f65bb-d26a-4dfd-894a-faed352f8474)


### Printing Y_prednew
![324681010-66ed7bc1-6b4b-4fea-a1e6-7bca7dad684b](https://github.com/gauthamkrishna7/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/141175025/b60bae34-0499-40b2-9318-cc634ed75dc3)

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

