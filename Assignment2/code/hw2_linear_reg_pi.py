import numpy as np
################SVD with a random 3x2 matrix

training_setA = []

def setup():
    for i in range(datasetSize):
        x1 = np.random.randint(-5, 5)
        x2 = np.random.randint(-5, 5)
        training_setA.append((x1, x2))


datasetSize = 3
setup()

U, d, VT = np.linalg.svd(training_setA)

print(' U: {} :  [d: {} , VT:{}]'.format(U.shape, d.shape, VT.shape))

D = np.diag(d)
print("D shape is ", D.shape)
D_inver = np.linalg.inv(D)
print(" D inverse shape {} : D shape :{}  \n ".format(D_inver.shape, D.shape))

Dplus = np.concatenate((D_inver, np.array([[0, 0]]).T), axis=1)

Aplus = np.dot(VT.T, np.dot(Dplus, U.T))
print("Inverse matrix shape is ", Aplus.shape)

# ----------verify
print(" A multiplies by A T : Should be an identity matrix \n ", np.dot(Aplus, training_setA))
print("Pseudo inverse not using np library: \n ", Aplus)
print("Pseudo inverse using np library: \n ", np.linalg.pinv(training_setA))

