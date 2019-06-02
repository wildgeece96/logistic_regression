import numpy as np 
import matplotlib.pyplot as plt 

from logistic_regression import *  


def truef(X): # define the true function  
    return 1.5* X[:, 0] -  X[:, 1] -1.0

# make sample data
X = np.random.randn(50, 2)
t = np.where(truef(X)>0.0, 1.0, 0.0)

# fittting 
logit = LogisticRegression(eta=1.0, num_iter=100)
logit.fit(X, t)

print("the modified weight ", logit.w)

# plot the result 

import matplotlib.pyplot as plt
# preparation 
seq = np.arange(-3, 3, 0.01)
X_1, X_2 = np.meshgrid(seq, seq)
X_1, X_2 = X_1.reshape(-1,1), X_2.reshape(-1,1)
X_all = np.concatenate([X_1,X_2],axis=1)
X_all = X_all.reshape(600,600, 2)
zlist = np.zeros([600,600])
for i in range(600):
    zlist[i,:] = logit.pred(X_all[i, :, :])

# plotting 
plt.figure(figsize=(12,8))
plt.imshow(zlist, extent=[-3,3,-3,3], origin='lower', cmap=plt.cm.PiYG_r)
plt.scatter(X[np.where(t==1)][:, 0],X[np.where(t==1)][:, 1], color='red', label='positive')
plt.scatter(X[np.where(t==0)][:, 0],X[np.where(t==0)][:, 1], color='blue', label='negative')
plt.legend()  
plt.show()