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

def error(trnY, y, lamda):
    err = 0    
    for i in range(len(trnY)):
    	if(y[i] < 0.00001):
    		y[i] = 0.00001
    	elif(y[i] > 0.99999):
    		y[i] = 0.99999
        err += trnY[i]*np.log(y[i])
        err += (1-trnY[i])*np.log(1-y[i])
    err = -1*err/len(trnY)
    # L2 regularization
    err += lamda * np.linalg.norm(w)**2
    # L1 regularization
    # err += lamda * np.sum(np.abs(w))
    return err

trnX = np.array([feat(trainX, i) for i in range(trainX.shape[0]) if trainY[i] == 2 or trainY[i] == 3])/256.0
trnY = np.array([1 if trainY[i] == 2 else 0 for i in range(trainX.shape[0]) if trainY[i] == 2 or trainY[i] == 3])
valIndices = np.random.choice(len(trnX), 2000)
nonValIndices = [x for x in range(len(trnX)) if x not in valIndices]

valX = trnX[valIndices]
valY = trnY[valIndices]

trnX = trnX[nonValIndices]
trnY = trnY[nonValIndices]

tstX = np.array([feat(testX, i) for i in range(testX.shape[0]) if testY[i] == 2 or testY[i] == 3])/256.0
tstY = np.array([1 if testY[i] == 2 else 0 for i in range(testX.shape[0]) if testY[i] == 2 or testY[i] == 3])

len_weight = {}
lam_acc = {}
lam_weights = {}
lamdas = [0.01, 0.001, 0.0001, 0.00001]

for lamda in lamdas:
	len_weight[lamda] = []

	print "For lamda = ", lamda
	w = np.random.rand(785)-0.5
	w_past = np.zeros((4, 785))
	prev_los = -1;
	count = 0
	trn_losses = []
	tst_losses = []
	val_losses = []
	trn_acc = []
	tst_acc = []
	val_acc = []

	lr = 0.001
	for i in range(1000):
		prediction = sigmoid(np.dot(valX, w))
		prediction[prediction > 0.5] = 1
		prediction[prediction <= 0.5] = 0
		los = error(valY, sigmoid(np.dot(valX, w)), lamda)
		val_losses.append(los)
		tst_losses.append(error(tstY, sigmoid(np.dot(tstX, w)), lamda))
		trn_losses.append(error(trnY, sigmoid(np.dot(trnX, w)), lamda))
		trn_acc.append(np.mean(np.abs(trnY - np.round(sigmoid(np.dot(trnX, w))))))
		tst_acc.append(np.mean(np.abs(tstY - np.round(sigmoid(np.dot(tstX, w))))))
		val_acc.append(np.mean(np.abs(valY - np.round(sigmoid(np.dot(valX, w))))))
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
		# L2
		grad += 2*lamda*w
		# L1
		# grad += 2*lamda*np.sign(w)
		w = w + lr*grad
		len_weight[lamda].append(np.linalg.norm(w))

	w_f = w_past[0]
	lam_weights[lamda] = w_f[1:].reshape(28, 28)
	tst_prediction = sigmoid(np.dot(tstX, w_f))
	tst_cost = error(tstY, sigmoid(np.dot(tstX, w_f)), lamda)
	tst_prediction[tst_prediction > 0.5] = 1
	tst_prediction[tst_prediction <= 0.5] = 0
	lam_acc[lamda] = (1-np.mean(np.abs(tstY - tst_prediction)))*100
	print "[Classification error = ", lam_acc[lamda],", Loss = ", tst_cost, "]"

	trn_acc = np.array(trn_acc)
	tst_acc = np.array(tst_acc)
	val_acc = np.array(val_acc)

# plt.plot([x for x in range(0, len(trn_losses))], trn_losses, label = "train")
# plt.plot([x for x in range(0, len(tst_losses))], tst_losses, label = "test")
# plt.plot([x for x in range(0, len(val_losses))], val_losses, label = "validation")
# plt.xlabel("Iterations")
# plt.ylabel("Entropy")
# plt.legend(loc='upper right', shadow=True)
# plt.show()

	plt.plot([x for x in range(0, len(trn_acc))], (1-trn_acc)*100, label = "train" + str(lamda))
	# plt.plot([x for x in range(0, len(tst_acc))], (1-tst_acc)*100, label = "test")
	# plt.plot([x for x in range(0, len(val_acc))], (1-val_acc)*100, label = "validation")

plt.xlabel("Iterations")
plt.ylabel("Classification Accuracy")
plt.legend(loc='lower right', shadow=True)

plt.show()

for key in len_weight.keys():
	plt.plot([x for x in range(len(len_weight[key]))], len_weight[key], label = "w_len," + str(key))

plt.xlabel("Iterations")
plt.ylabel("w_len")
plt.legend(loc='lower right', shadow=True)
plt.show()

# plt.plot([np.log10(x) for x in lam_acc.keys()], [lam_acc[key] for key in lam_acc.keys()])
plt.scatter([np.log10(x) for x in lam_acc.keys()], [lam_acc[key] for key in lam_acc.keys()])
plt.xlabel("Log Lamda")
plt.ylabel("Classification Accuracy")
plt.show()

for lam in lamdas:
	plt.imshow(lam_weights[lam], cmap = 'Greys')
	plt.title(str(lam))
	plt.show()

# plt.imshow(eightimg, cmap= 'Greys')
# plt.show()