import numpy as np
import random

# "pip install python-mnist"
from mnist import MNIST
mndata = MNIST('Dataset')
trainX = np.array(mndata.load_training()[0])[:20000]
trainY = np.array(mndata.load_training()[1])[:20000]
testX = np.array(mndata.load_testing()[0])[:2000]
testY = np.array(mndata.load_testing()[1])[:2000]

# Logistic Regression Portion
def sigmoid(x):
    return 1.0/(1+np.exp(-1*np.array(x)))

def feat(id):
    return [1] + trainX[id].tolist()

def error(trnY, y):
    err = 0    
    for i in range(len(trnY)):
    	if(y[i] < 0.00001):
    		y[i] = 0.00001
    	elif(y[i] > 0.99999):
    		y[i] = 0.99999
        err += trnY[i]*np.log(y[i])
        err += (1-trnY[i])*np.log(1-y[i])
    return err/len(trnY)


trnX = np.array([feat(i) for i in range(trainX.shape[0]) if trainY[i] == 2 or trainY[i] == 3])/256.0
trnY = np.array([1 if trainY[i] == 2 else 0 for i in range(trainX.shape[0]) if trainY[i] == 2 or trainY[i] == 3])
valIndices = np.random.choice(len(trnX), 2000)
valX = trnX[valIndices]
valY = trnY[valIndices]
nonValIndices = [x for x in range(len(trnX)) if x not in valIndices]
trnX = trnX[nonValIndices]
trnY = trnY[nonValIndices]
tstX = np.array([feat(i) for i in range(testX.shape[0]) if testY[i] == 2 or testY[i] == 3])/256.0
tstY = np.array([1 if testY[i] == 2 else 0 for i in range(testX.shape[0]) if testY[i] == 2 or testY[i] == 3])


w = np.random.rand(785)-0.5
print  "-1, Test::", error(tstY, sigmoid(np.dot(tstX, w)))

lr = 0.0001
for i in range(15):
    y = sigmoid(np.dot(trnX, w))
    grad = np.dot((trnY-y), trnX)
    w = w + lr*grad
    b = sigmoid(np.dot(trnX, w))
    print np.linalg.norm(tstY - sigmoid(np.dot(tstX, w))),"::", error(tstY, sigmoid(np.dot(tstX, w)))

