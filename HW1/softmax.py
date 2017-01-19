import numpy as np
import random
from mnist import MNIST
import matplotlib.pyplot as plt

mndata = MNIST('Dataset')
trainX = np.array(mndata.load_training()[0])[:20000]
trainY = np.array(mndata.load_training()[1])[:20000]

#random selection of data for training and validation
valIndices = np.random.choice(len(trainX), 2000)
nonValIndices = [x for x in range(len(trainX)) if x not in valIndices]

valX = trainX[valIndices]
valY = trainY[valIndices]

trainX = trainX[nonValIndices]
trainY = trainY[nonValIndices]

testX = np.array(mndata.load_testing()[0])[:2000]
testY = np.array(mndata.load_testing()[1])[:2000]

def feat(data,i):
    return [1] + data[i].tolist()

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

def softmax(x):
    x = np.exp(x)
    x = x/x.sum(axis=1)[:, None]
    return x

def error(trnY, y, lamda):
    err = 0    
    for i in range(len(trnY)):
        for k in range(10):
            if(y[i, k] < 0.00001):
                y[i, k] = 0.00001
            elif(y[i, k] > 0.99999):
                y[i, k] = 0.99999
            err += trnY[i, k]*np.log(y[i, k])
    err=-1*err/len(trnY)
    # L2 regularization
    err += lamda * np.linalg.norm(w)**2
    # L1 regularization
    # err += lamda * np.sum(np.abs(w))
    return err

def accuracy(y, y_):
    y_ = np.argmax(y_, axis = 1)
    y = np.argmax(y, axis = 1)
    count = 0
    for i in range(len(y_)):
        if y_[i] == y[i]:
            count += 1
    return count*100.0/len(y_)

err = [[], [], []]
acc = [[], [], []]

lr = 0.0001
w = (np.random.rand(10, 785)-0.5)
w_past = np.zeros((4, 10, 785))
count = 0
past_loss = 0

lamda = [0.01,0.001, 0.0001, 0.00001]

max_acc = 0

#L1 Regularization
err1 = [[], [], []]
acc1 = [[], [], []]
best_lamda = 0
for l in lamda:
    err = [[], [], []]
    acc = [[], [], []]

    lr = 0.0001
    w = (np.random.rand(10, 785)-0.5)
    w_past = np.zeros((4, 10, 785))
    count = 0
    past_loss = 0
    for i in range(300):
        print i
        y = softmax(np.dot(trnX,np.transpose(w)))
        loss = error(trnY, y, l)
        err[0].append(loss)
        err[1].append(error(tstY, softmax(np.dot(tstX,np.transpose(w))),l))
        err[2].append(error(valY, softmax(np.dot(valX,np.transpose(w))),l))
        acc[0].append(accuracy(trnY, y))
        acc[1].append(accuracy(tstY, softmax(np.dot(tstX,np.transpose(w)))))
        acc[2].append(accuracy(valY, softmax(np.dot(valX,np.transpose(w)))))
        if past_loss <= loss and i > 78:
            count += 1
            w_past[0] = w_past[1]
            w_past[1] = w_past[2]
            w_past[2] = w_past[3]
            w_past[3] = w
        elif past_loss > loss and i > 10:
            count = 0
            past_loss = -1
        if count > 3 and i > 10:
            break
        grad = np.dot(np.transpose(trnY-y), trnX)
        # L2
        grad += 2*l*w
        # L1
#         grad += l*np.sign(w)
        w = w + lr*grad
        
    w_f = w_past[0]
    if acc[2][-3] > max_acc:
        best_lamda = l
        max_acc = acc[2][-3]
        w_best = w_f
        err1 = err
        acc1 = acc

w_f = w_past[0]
print "Final testing error = ", err[1][-3]
print "Final training error = ", err[0][-3]
print "Final validation error = ", err[2][-3]

plt.plot([x for x in range(len(err[0]))], err[0], label = "train")
plt.plot([x for x in range(len(err[1]))], err[1], label = "test")
plt.plot([x for x in range(len(err[2]))], err[2], label = "validation")
plt.xlabel("Iterations")
plt.ylabel("Entropy")
plt.title("Using L2 with Lamda = " + str(best_lamda))
plt.legend(loc='upper right', shadow=True)
plt.show()

plt.plot([x for x in range(len(acc[0]))], acc[0], label = "train")
plt.plot([x for x in range(len(acc[1]))], acc[1], label = "test")
plt.plot([x for x in range(len(acc[2]))], acc[2], label = "validation")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Using L2 with Lamda = " + str(best_lamda))
plt.legend(loc='lower right', shadow=True)
plt.show()

plt.imshow(w_best[1][1:].reshape(28, 28))
plt.show()
