import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


################SVD with a random 3x2 matrix
A = np.array([[-1,2],[3,-2],[5,7]])
U,d,VT = np.linalg.svd(A)

print(' U: {} :  [d: {} , VT:{}]'.format(U.shape, d.shape, VT.shape ))


#print(" U is \n" , U)
#print(" d is \n " , d)
#print(" V transpose is \n" , VT.T)
#print(" V  is \n" , VT.T)
#print(" U transpose is \n" , U.T)
D  =  np.diag(d)
print("D shape is ", D.shape)
D_inver = np.linalg.inv(D)
print(" D inverse shape {} : D shape :{}  \n ".format(D_inver.shape, D.shape))

## D inverse is 2x2 , have to make it 2x3
Dplus = np.concatenate((D_inver, np.array([[0,0]]).T),axis = 1)
#print(" d plus \n ", Dplus.shape, Dplus)
#print ("#####################")
#print (VT.T , "\n", Dplus , "\n", U.T)
#print ("#####################")
Aplus = np.dot(VT.T, np.dot(Dplus,U.T))
print("Inverse matrix shape is " , Aplus.shape)
#print("Inverse matrix  is \n" , Aplus)
#----------verify
print(" A multiplies by A T : Should be an identity matrix \n ", np.dot(Aplus, A))
#print (" V shape is ", VT.T)

print("Pseudo inverse not using np library: \n ", Aplus)
print("Pseudo inverse using np library: \n ", np.linalg.pinv(A))

########## END OF Sample matrix

################ Housing data

#Read housing data, skip the heading row
df_housing = pd.read_csv('house_data.csv', skiprows=1)
df_housing.head()
#A = df.iloc[:, 1:]
#df['Bias'] = 1
df_housing.shape

print(df_housing.shape)

#exit(0)
U_housing,d_housing,VT_housing = np.linalg.svd(df_housing)

#print(" U is \n" , U)
#print(" d is \n " , d)
#print(" V transpose is \n" , VT.T)
#print(" V  is \n" , VT.T)
#rint(" U transpose is \n" , U.T)
D_housing  =  np.diag(d_housing)
print(' U: {} :  [D: {} , VT:{}]'.format(U_housing.shape, D_housing.shape, VT_housing.shape ))
#print("D shape is \n ", D.shape)
D_housing_inver = np.linalg.inv(D_housing)
print(" D inverse shape" , D_housing_inver.shape)

Dplus_housing = D_housing_inver

for i in range (21566):

    Dplus_housing = np.concatenate((Dplus_housing, np.array([[0,0,0,0,0,0,0,0]]).T),axis = 1)


print(" d plus shape : ", Dplus_housing.shape)

print ("#####################")
Aplus_housing = np.dot(VT_housing.T, np.dot(Dplus_housing,U_housing.T))


#print("Inverse matrix shape is {} , V shape is {} " , format(Aplus.shape, VT.T.shape))

#----------verify
print(" A multiplies by A T : Should be an identity matrix\n ", np.dot(Aplus_housing, df_housing))


print ("------------Verify ----------")
print("Pseudo inverse not using np library: \n ", Aplus_housing)
print("Pseudo inverse using np library: \n ", np.linalg.pinv(df_housing))


