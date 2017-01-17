
# coding: utf-8

# In[1]:

import numpy as np
import random
from mnist import MNIST
import matplotlib.pyplot as plt


# In[143]:

mndata = MNIST('Dataset')
trainX = np.array(mndata.load_training()[0])[:18000]
trainY = np.array(mndata.load_training()[1])[:18000]
valX = np.array(mndata.load_training()[0])[18000:20000]
valY = np.array(mndata.load_training()[1])[18000:20000]
testX = np.array(mndata.load_testing()[0])[:2000]
testY = np.array(mndata.load_testing()[1])[:2000]


# In[144]:

def feat(data,i):
    return [1] + data[i].tolist()


# In[145]:

trnX = np.array([feat(trainX,i) for i in range(trainX.shape[0])])/256.0
tesX = np.array([feat(testX,i) for i in range(testX.shape[0])])/256.0
vaX = np.array([feat(valX,i) for i in range(valX.shape[0])])/256.0


# In[146]:

trnY = np.zeros((18000,10))
tesY = np.zeros((2000,10))
vaY = np.zeros((2000,10))
for i in range(18000):
    trnY[i, trainY[i]]=1;
    if i<2000:
        vaY[i, valY[i]]=1;
        tesY[i, testY[i]]=1;


# In[147]:

def softmax(x):
    x=np.exp(x)
    x=x/x.sum(axis=1)[:,None]
    return x


# In[148]:

def error(trnY, y):
    er = 0    
    for i in range(len(trnY)):
        for k in range(10):
            if(y[i, k] < 0.00001):
                y[i, k] = 0.00001
            elif(y[i, k] > 0.99999):
                y[i, k] = 0.99999
            er += trnY[i, k]*np.log(y[i, k])
    return -1*er/len(trnY)


# In[155]:

y=np.zeros((18000, 10))
su=np.zeros(18000)
err=np.zeros((100, 3))


# In[ ]:




# In[156]:

lr=0.001


# In[157]:

w = (np.random.rand(10, 785)-0.5)/1000


# In[158]:

for iter in range (100):
    y=softmax(np.dot(trnX,np.transpose(w)))
    err[iter,0]=error(trnY, y)
    err[iter,1]=error(tesY, softmax(np.dot(tesX,np.transpose(w))))
    err[iter,2]=error(vaY, softmax(np.dot(vaX,np.transpose(w))))
    grad=np.dot(np.transpose(trnY-y), trnX)
    w=w+lr*grad
    


# In[159]:

plt.plot([x for x in range(0, len(err))], err[:,0], label = "train")
plt.plot([x for x in range(0, len(err))], err[:,1], label = "test")
plt.plot([x for x in range(0, len(err))], err[:,2], label = "validation")
plt.legend(loc='upper right', shadow=True)
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:



