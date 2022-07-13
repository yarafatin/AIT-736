import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


################SVD with a random 3x2 matrix

training_setA=[]

def setup():
    for i in range(datasetSize):
        x1 = np.random.randint(-5, 5)
        x2 = np.random.randint(-5, 5)
        training_setA.append((x1, x2))

datasetSize=3

setup()

print (training_setA)

U,d,VT = np.linalg.svd(training_setA)

print(' U: {} :  [d: {} , VT:{}]'.format(U.shape, d.shape, VT.shape ))

D  =  np.diag(d)
print("D shape is ", D.shape)
D_inver = np.linalg.inv(D)
print(" D inverse shape {} : D shape :{}  \n ".format(D_inver.shape, D.shape))

Dplus = np.concatenate((D_inver, np.array([[0,0]]).T),axis = 1)

Aplus = np.dot(VT.T, np.dot(Dplus,U.T))
print("Inverse matrix shape is " , Aplus.shape)

#----------verify
print(" A multiplies by A T : Should be an identity matrix \n ", np.dot(Aplus, training_setA))
print("Pseudo inverse not using np library: \n ", Aplus)
print("Pseudo inverse using np library: \n ", np.linalg.pinv(training_setA))

########## END OF Sample matrix

################ Housing data
print ("---------------- START OPTION 2 - housing dataset ")
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


