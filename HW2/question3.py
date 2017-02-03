
# coding: utf-8

# In[59]:

import numpy as np
import random
from mnist import MNIST
import matplotlib.pyplot as plt


# In[58]:

mndata = MNIST('../HW1/Dataset')
trainX = np.array(mndata.load_training()[0])[:50000]
trainY = np.array(mndata.load_training()[1])[:50000]

testX = np.array(mndata.load_testing()[0])[:1000]
testY = np.array(mndata.load_testing()[1])[:1000]


# In[60]:

valIndices = np.random.choice(len(trainX), 2000)
nonValIndices = [x for x in range(len(trainX)) if x not in valIndices]

valX = trainX[valIndices]
valY = trainY[valIndices]

trainX = trainX[nonValIndices]
trainY = trainY[nonValIndices]

testX = np.array(mndata.load_testing()[0])[:2000]
testY = np.array(mndata.load_testing()[1])[:2000]

def feat(data,i):
    return data[i].tolist()

def oneHot(clas, noOfClasses):
    feat = np.zeros(noOfClasses)
    feat[clas] = 1;
    return feat

trnX = np.array([feat(trainX,i) for i in range(trainX.shape[0])])/256.0
trnY = np.array([oneHot(trainY[i], 10) for i in range(trainX.shape[0])])

tstX = np.array([feat(testX,i) for i in range(testX.shape[0])])/256.0
tstY = np.array([oneHot(testY[i], 10) for i in range(testX.shape[0])])

valX = np.array([feat(valX,i) for i in range(valX.shape[0])])/256.0
valY = np.array([oneHot(valY[i], 10) for i in range(valX.shape[0])])


# In[63]:

def sigmoid(x):
    return 1.0/(1+np.exp(-1*x))

def lecun(x):
    return 1.7159*np.tanh(2.0*x/3)

def grad(x):
    return x*(1-x)

def gradLecun(x):
    t = 2.0*x/3
    return 1.7159*(1-t**2)

def softmax(x):
    x = np.exp(x)
    x = x/x.sum(axis=1)[:, None]
    return x
    
maxValAcc = 0
lr = 0.00001
trnAcc = []
valAcc = []
tstAcc = []

# randomly initialize our weights with mean 0
n_hid_1 = 100
fan_in = 1.0#/np.sqrt(784)
# print fan_in
W1 = fan_in*(2*np.random.random((784,n_hid_1)) - 1)
bias = fan_in*(2*np.random.random((n_hid_1)) - 1)
fan_in_h = 1.0#/np.sqrt(n_hid_1)
W2 = fan_in_h*(2*np.random.random((n_hid_1,10)) - 1)
bias2 = fan_in_h*(2*np.random.random((10)) - 1)

for j in xrange(300):
    indices = np.random.choice(len(trnX), len(trnX))
    tempY = trnY[indices]
    tempX = trnX[indices]
    A1 = np.dot(tempX, W1) + bias
    l1 = sigmoid(A1)
    
    A2 = np.dot(l1, W2) + bias2
    l2 = softmax(A2)

    # Errors in output layer
    d2 = (l2 - tempY)
    dbias2 = np.sum(d2, axis = 0)
        
    # Delta of W2
    dW2 = np.dot(l1.T, d2)

    # Errors in 1st hidden layer
    d1 = np.dot(d2, W2.T)*grad(l1)
    dbias = np.sum(d1, axis = 0)    
    
    # Delta W2
    dW1 = np.dot(tempX.T, d1)
    
    W2 -= lr*dW2
    bias2 -= lr*dbias2
    W1 -= lr*dW1
    bias -= lr*dbias
    
    prediction = softmax(np.dot(sigmoid(np.dot(valX, W1)+bias), W2)+bias2)
    correct = [1 if a == b else 0 for (a, b) in zip(np.argmax(valY, axis = 1), np.argmax(prediction, axis = 1))]
    valAcc.append(np.sum(correct)*100.0/len(valX))
    
    prediction = softmax(np.dot(sigmoid(np.dot(trnX, W1)+bias), W2)+bias2)
    correct = [1 if a == b else 0 for (a, b) in zip(np.argmax(trnY, axis = 1), np.argmax(prediction, axis = 1))]
    trnAcc.append(np.sum(correct)*100.0/len(trnX))   
    
    prediction = softmax(np.dot(sigmoid(np.dot(tstX, W1)+bias), W2)+bias2)
    correct = [1 if a == b else 0 for (a, b) in zip(np.argmax(tstY, axis = 1), np.argmax(prediction, axis = 1))]
    tstAcc.append(np.sum(correct)*100.0/len(tstX))   
    
    print tstAcc[-1]


# In[64]:

print "Final test accuracy = ", tstAcc[-1]
print "Final training accuracy =", trnAcc[-1]
print "Final validation accuracy = ", valAcc[-1]
plt.plot([x+1 for x in range(len(trnAcc))], trnAcc, label = 'Training Accuracy')
plt.plot([x+1 for x in range(len(valAcc))], valAcc, label = 'Validation Accuracy')
plt.plot([x+1 for x in range(len(tstAcc))], tstAcc, label = 'Testing Accuracy')
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend(loc='lower right', shadow=True)
plt.show()


# In[ ]:



