import numpy as np 

def sigmoid(x):
    return 1./(1.+np.exp(-x))

class LogisticRegression():
    
    def __init__(self, eta=0.1, num_iter=50):
        self.eta = eta 
        self.num_iter = num_iter
    
    def fit(self, X, t):
        """
        inputs: 
            X : 2-d array. shape: [N, d].
                N : the number of data.  
                d:  the number of features.
            t : 1-d array. shape: [N].
                the desired values
        """
        n_features = X.shape[1]+1
        X = np.append(X, np.ones(shape=[X.shape[0],1]),axis=1)
        self.w = np.random.randn(1,n_features)
        # predict the value
        for i in range(self.num_iter):
            y_hat = sigmoid(np.dot(X, self.w.reshape(-1,1)))  # --> [N, 1]
            grad = self.calculate_grad(X, y_hat, t)
            self.w -= self.eta * grad 
    
    def calculate_grad(self, X, y_hat, t):
        d_y_hat = 2*(y_hat - t.reshape(-1,1)) # -> [N, 1]
        d_xomega = (1 - y_hat)* y_hat  # -> [N, 1]
        d_omega = X # -> [N, d]
        return np.dot((d_y_hat*d_xomega).T, d_omega) 
        
    def pred(self, X):
        X = np.append(X, np.ones(shape=[X.shape[0],1]),axis=1)
        return sigmoid(np.dot(X, self.w.reshape(-1,1))).flatten()
        
        
        
        
        
        