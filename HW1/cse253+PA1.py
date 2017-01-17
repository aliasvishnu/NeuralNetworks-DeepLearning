
# coding: utf-8

# In[12]:

import numpy as np
import random
from mnist import MNIST


# In[47]:

mndata = MNIST('Dataset')
trainX = np.array(mndata.load_training()[0])[:18000]
trainY = np.array(mndata.load_training()[1])[:18000]
valX = np.array(mndata.load_training()[0])[18000:20000]
valY = np.array(mndata.load_training()[1])[18000:20000]
testX = np.array(mndata.load_testing()[0])[:2000]
testY = np.array(mndata.load_testing()[1])[:2000]


# In[49]:

trainX.shape


# In[58]:

valX.shape[0]


# In[62]:

def feat1(i):
    return [1] + trainX[i].tolist()
def feat2(i):
    return [1] + testX[i].tolist()
def feat3(i):
    return [1] + valX[i].tolist()


# In[65]:

trnX = np.array([feat1(i) for i in range(trainX.shape[0])])/256.0
tesX = np.array([feat2(i) for i in range(testX.shape[0])])/256.0
vaX = np.array([feat3(i) for i in range(valX.shape[0])])/256.0


# In[75]:

w = np.random.rand(10, 785)/1000


# In[165]:

arr=np.zeros((18000, 10))
su=np.zeros(18000)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[166]:

for iter in range (1):
    arr=np.dot(w,np.transpose(trnX))
    arr=np.exp(arr)
    su=arr.sum(axis=0)
    arr=arr/su
        


# In[168]:

arr[0]


# In[ ]:




# In[ ]:




# In[ ]:



