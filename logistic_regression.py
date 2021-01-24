import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, lr=0.001, verbose=False, seed=42):
        
        """

        Arguments:
        input_dim -- number of features, int
        lr -- learning_rate
        verbose -- bool, True if it's needed to output accuracy while fitting
        
        """
        
        super(LogisticRegression, self).__init__()
        
        self.lr = lr
        self.verbose = verbose
        torch.manual_seed(seed)
        
        self.W = torch.zeros((input_dim, 1), dtype=torch.float32)
        self.b = torch.zeros((1, 1), dtype=torch.float32)
        torch.nn.init.normal_(self.W, mean=0, std=0.1)

    def propagate(self, X, Y):
        """
        forward + backward

        Arguments:
        X -- sample of size (m, input_dim)
        Y  -- labels of size (m, 1)

        Return:
        cost -- cross entropy loss for logistic regression
        dw -- gradient of the loss with respect to w
        db -- gradient of the loss with respect to b
        
        """
    
        m = X.shape[0]
        
        A = torch.sigmoid(X @ self.W + self.b)            
        cost = torch.nn.functional.binary_cross_entropy(A, Y)
        
        # grad of cost + grad of L2 regularization
        dw = X.T @ (A - Y) / m + self.W
        db = torch.mean(A - Y)

        cost = torch.squeeze(cost)

        grads = {"dw": dw,
                 "db": db}

        return grads, cost
    
    def fit(self, X, Y, num_iter=1000, batch_size=32):
        
        """
        function for training

        Arguments:
        X -- sample of size (m, input_dim)
        Y  -- labels of size (m)
        num_iter -- number of iterations of SGD descent
        batch_size -- size of one batch
        
        """
        
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        Y = torch.reshape(Y, (-1, 1))
        
        m = X.shape[0]
        
        for i in range(num_iter):
            perm = torch.randperm(m)
            for j in range(m // batch_size):
                idx = perm[j * batch_size:(j+1) * batch_size]
                X_sub = X[idx]
                Y_sub = Y[idx]
                
                grads, cost = self.propagate(X_sub, Y_sub)
                
                dw = grads['dw']
                db = grads['db']
                
                self.W -= self.lr * dw
                self.b -= self.lr * db
            
            if i % 100 == 0 and self.verbose:
                _, cost = self.propagate(X, Y)
                Y_pred = self.predict(X)
                Y_pred = (Y_pred > 0.5).type(torch.float32)
                print('accuracy is %.2f ' % (1 - torch.mean(torch.abs(Y_pred - Y)).item()))
    
    def predict(self, X):
        
        """
        function for predicting logits

        Arguments:
        X -- sample of size (m, input_dim)
        
        Return:
        A -- logits for sample X
        
        """
        
        X = torch.tensor(X, dtype=torch.float32)
        return torch.sigmoid(X @ self.W + self.b)