import numpy as np
import random
import matplotlib.pyplot as plt

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

def feat(data, id):
    return [1] + data[id].tolist()

def error(trnY, y):
    err = 0    
    for i in range(len(trnY)):
    	if(y[i] < 0.00001):
    		y[i] = 0.00001
    	elif(y[i] > 0.99999):
    		y[i] = 0.99999
        err += trnY[i]*np.log(y[i])
        err += (1-trnY[i])*np.log(1-y[i])
    return -1*err/len(trnY)


valIndices = np.random.choice(len(trnX), 2000)
nonValIndices = [x for x in range(len(trnX)) if x not in valIndices]

trnX = np.array([feat(trainX, i) for i in range(trainX.shape[0]) if trainY[i] == 2 or trainY[i] == 8])/256.0
trnY = np.array([1 if trainY[i] == 2 else 0 for i in range(trainX.shape[0]) if trainY[i] == 2 or trainY[i] == 8])

valX = trnX[valIndices]
valY = trnY[valIndices]

trnX = trnX[nonValIndices]
trnY = trnY[nonValIndices]

tstX = np.array([feat(testX, i) for i in range(testX.shape[0]) if testY[i] == 2 or testY[i] == 8])/256.0
tstY = np.array([1 if testY[i] == 2 else 0 for i in range(testX.shape[0]) if testY[i] == 2 or testY[i] == 8])

w = np.random.rand(785)-0.5
w_past = np.zeros((4, 785))
prev_los = -1;
count = 0
trn_losses = []
tst_losses = []
val_losses = []

lr = 0.001
for i in range(10000):
	prediction = sigmoid(np.dot(valX, w))
	prediction[prediction > 0.5] = 1
	prediction[prediction <= 0.5] = 0
	los = error(valY, sigmoid(np.dot(valX, w)))
	val_losses.append(los)
	tst_losses.append(error(tstY, sigmoid(np.dot(tstX, w))))
	trn_losses.append(error(trnY, sigmoid(np.dot(trnX, w))))
	print "[Classification error = ", np.mean(np.abs(valY - prediction)),", Loss = ", los, "]"
	if los >= prev_los:
		prev_los = los
		count += 1;
		w_past[0] = w_past[1]
		w_past[1] = w_past[2]
		w_past[2] = w_past[3]
		w_past[3] = w
	else:
		prev_los = -1;
		count = 0;
	if count > 3:
		break
	y = sigmoid(np.dot(trnX, w))
	grad = np.dot((trnY-y), trnX)
	w = w + lr*grad

w_f = w_past[3]
tst_prediction = sigmoid(np.dot(tstX, w_f))
tst_cost = error(tstY, sigmoid(np.dot(tstX, w_f)))
tst_prediction[tst_prediction > 0.5] = 1
tst_prediction[tst_prediction <= 0.5] = 0
print "[Classification error = ", np.mean(np.abs(tstY - tst_prediction)),", Loss = ", tst_cost, "]"

plt.plot([x for x in range(0, len(trn_losses))], trn_losses, label = "train")
plt.plot([x for x in range(0, len(tst_losses))], tst_losses, label = "test")
plt.plot([x for x in range(0, len(val_losses))], val_losses, label = "validation")
plt.legend(loc='upper right', shadow=True)
plt.show()