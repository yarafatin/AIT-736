import random
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




datasetSize=100
training_set=[]


def setup():
    for i in range(datasetSize):
        x1 = np.random.randint(-100, 100)
        x2 = np.random.randint(-100, 100)
        training_set.append(((x1, x2), isPositiveTrainingY(x1, x2)))


def isPositiveTrainingY(x1, x2):
    if (x1+x2)<0:
        return -1
    else:
        return 1

#Activation function
def activation_function(x, weights, b):
    value = sum(x * weights) + b
    if value<0:
        return -1
    else:
        return 1

setup()

print (type(training_set), training_set)

#print(training_set)
#print(weights)
errors = []
eta = .5
epoch = 30
b = 0

x1 = [training_set[i][0][0] for i in range(datasetSize)]
x2 = [training_set[i][0][1] for i in range(datasetSize)]
y = [training_set[i][1] for i in range(datasetSize)]
y1=y
df = pd.DataFrame(
    {'x1': x1,
     'x2': x2,
     'y': y
     })



#plt.scatter(x1, x2, marker=y)
#plt.grid()
sns.lmplot(x="x1", y="x2", data=df, hue='y', fit_reg=False, markers=["<", ">"], legend=True,
           palette="Set1", ).fig.suptitle("Classification of traing data with random values")

errors = []
eta = .5
epoch = 30
b = 0

mydict= {}

weights = np.random.rand(2)
b = 0
for i in range(epoch):
    errorcount = 0
    for x, y in training_set:
        error = y - activation_function(x, weights, b)
        if (error != 0):
            errorcount = errorcount + 1
        errors.append(error)
        for index, value in enumerate(x):
            # print(w[index])
            weights[index] += eta * error * value/100
            b += eta * error
    print(' Iteration: {} :  [errorcount: {} , weights:{}, bias:{} ]'.format(i+1, errorcount, weights, b ))
    mydict[i+1]=errorcount

#print(mydict)

names = list(mydict.keys())
values = list(mydict.values())
plt.figure(4)
plt.bar(range(len(mydict)), values, tick_label=names)
plt.xticks(rotation = 90)
plt.xlabel('iteration')
plt.ylabel('errorcount')
plt.title(" Count of errors in different iteration")
plt.show(block=True)


output = []
for x in errors:
    if x not in output:
        output.append(x)
#print(output)
#print(len(errors))
from collections import Counter
d = Counter(errors)

print('{} has occurred {} times'.format(2, d[2]))
print('{} has occurred {} times'.format(-2, d[-2]))
print('{} has occurred {} times'.format(0, d[0]))

a = [0,-b/weights[1]]
c = [-b/weights[0],0]
fig, ax = plt.subplots()
print(a, b, c, weights)
plt.plot(a,c)

#ax = plt.axes()
#ax.scatter(x1, x2, c = y1)
#ax.set_xlabel('X1')
#ax.set_ylabel('X2')
plt.title("Decision Boundary")
plt.show(block=True)

plt.figure("Error Value and Count")
plt.ylim([-3,3])
plt.plot(errors)
plt.title("Error value and count")
plt.show(block=True)


#plt.figure(figsize=(4, 4))
#ax = plt.axes()
#ax.scatter(x1, x2, c = y1)
##ax.set_xlabel('X1')
#ax.set_ylabel('X2')
#plt.show(block=True)