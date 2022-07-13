import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv('house_data.csv')

df.head()

df['Bias'] = 1
df.shape

unorganized = df.sample(frac=1, random_state=1)
unorganized.shape

train = unorganized.iloc[:int(0.7*len(unorganized)), :]
train.head(10)

print(train.shape)

test = unorganized.iloc[int(0.7*len(unorganized)): , :]
test.head(10)

A = train.iloc[:, 1:]
A.head()

#Pseudoinverse of A
A = np.array(A)
print(A.shape)
A_dag = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
print(A_dag.shape)

y = train.iloc[:, :1]
y.head()

y = np.array(y)
y.shape

print(A_dag.shape, y.shape)

thetas = np.matmul(A_dag, y)
thetas

X = np.array(test.iloc[:, 1:])
y = np.array(test.iloc[:, :1])
print(X.shape, y.shape)

print(X.shape, thetas.shape)

y_hat = np.round(np.matmul(X, thetas))
y_hat.shape

#for i in range(len(y_hat)):
 #   print(y[i], y_hat[i])

def checkMSE(y, y_hat):
    error = np.sum((y-y_hat)**2)/len(y)
    return error

print(" MSE is {} " , format(checkMSE(y, y_hat)))

plt.scatter(x=y, y=np.array(y), color='blue')
plt.scatter(x=y_hat, y=np.array(y), color='black')

plt.show()

#-------------------
import numpy as np
M = np.array([[1,2,0],[8,1,9],[2,3,1]])
Minv = np.linalg.inv(M)

print(Minv)

################SVD
A = np.array([[-1,2],[3,-2],[5,7]])


print(A)
U,d,VT = np.linalg.svd(A)

print(" U is \n" , U)
print(" d is \n " , d)
print(" V transpose is \n" , VT.T)
print(" V  is \n" , VT.T)
print(" U transpose is \n" , U.T)
D  =  np.diag(d)
print("D shape is \n ", D.shape)
D_inver = np.linalg.inv(D)
print(" D inverse\n " , D_inver.shape, D_inver , D)
Dplus = np.concatenate((D_inver, np.array([[0,0]]).T),axis = 1)
print(" d plus \n ", Dplus.shape, Dplus)
print ("#####################")
print (VT.T , "\n", Dplus , "\n", U.T)
print ("#####################")
Aplus = np.dot(VT.T, np.dot(Dplus,U.T))


print("Inverse matrix shape is " , Aplus.shape)
print("Inverse matrix  is \n" , Aplus)
#----------verify
print(" A multiplies by A T \n ", np.dot(Aplus, A))


print (" V shape is ", VT.T)

print(np.linalg.pinv(A))