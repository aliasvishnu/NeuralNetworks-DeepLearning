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



# Begin softmax here, do not change trainX, trainY, testY, testX
#okay BOSS :*
