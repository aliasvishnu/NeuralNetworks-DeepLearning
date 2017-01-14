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
    return 1/1+np.exp(-1*x)

def feat(id):
    return [1] + trainX[id].tolist()

def error(trnY, y):
    err = 0
    for i in range(len(trnY)):
        err += trnY[i]*math.log(y[i]) if y[i] != 0 else 0
        err += (1-trnY[i])*math.log(1-y[i]) if y[i] != 1 else 0
    return err/len(trnY)


trnX = np.array([feat(i) for i in range(trainX.shape[0]) if trainY[i] == 2 or trainY[i] == 3])/256.0
trnY = np.array([trainY[i]-2 for i in range(trainX.shape[0]) if trainY[i] == 2 or trainY[i] == 3])

w = np.random.rand(785)/1000

lr = 0.001
for i in range(100):
    y = sigmoid(np.dot(trnX, w))
    grad = np.dot((trnY-y), trnX)
    w = w - grad
    print i, "::", error(trnY, y)

print trnY != y

# Begin softmax here, do not change trainX, trainY, testY, testX

