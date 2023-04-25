# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.
## Program:


Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Thenmozhi P
RegisterNumber: 21222123116 
~~~
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter = ',')
X = data[:,[0,1]]
y = data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()


plt.show()

    return 1 / (1 + np.exp(-z))
    
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y,np.log(1-h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J,grad
    
X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)), X))
theta = np.array([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J
def gradient(theta,X,y):
    h = sigmoid(np.dot(X,theta))
    grad = np.dot(X.T,h-y)/X.shape[0]
    return grad
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta  = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y),method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min, x_max = X[:, 0].min() - 1,X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1,X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max, 0.1),np.arange(y_min,y_max, 0.1))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot = np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)),X))
    prob = sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X) == y)
~~~
## Output:

## Array value of x

![image](https://user-images.githubusercontent.com/95198708/234183968-c85d8559-54b9-435c-a834-2f839eb78af7.png)

## Array Value of y

![image](https://user-images.githubusercontent.com/95198708/234184044-1b6da0ea-807d-4e7c-879d-e8a72ef250a9.png)

## Exam 1- Score graph

![image](https://user-images.githubusercontent.com/95198708/234184099-4de7ecf8-dc50-4315-b21f-db1044b27a68.png)

## Sigmoid Function Graph

![image](https://user-images.githubusercontent.com/95198708/234184151-0c289c6c-fdd9-4e43-9729-8c7579dbeb97.png)

## X_train_grad value

![image](https://user-images.githubusercontent.com/95198708/234184204-c62a0600-af39-4812-be3e-2447d19ac807.png)

## Y_train_grad value

![image](https://user-images.githubusercontent.com/95198708/234184252-4271c798-d01a-4642-9a63-3026ca891a29.png)

## Print res.x

![image](https://user-images.githubusercontent.com/95198708/234184295-ddbb604c-026e-40ca-b675-60b2e4f4948e.png)

## Decision Boundary grapg for Exam Score


![image](https://user-images.githubusercontent.com/95198708/234184366-eafecc90-82ab-4341-89aa-907444c3fd67.png)

## Probability value

![image](https://user-images.githubusercontent.com/95198708/234184456-d82c432a-b41d-488a-b3c2-2beae67398b1.png)

## Prediction value of mean

![image](https://user-images.githubusercontent.com/95198708/234184592-232219c1-6725-44a4-868d-660c9b160e4a.png)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

