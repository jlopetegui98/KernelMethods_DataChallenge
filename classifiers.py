import numpy as np
from scipy import optimize

class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        print("Building K...")
        K = self.kernel(X,X)
        print("Building K - done")
        # Lagrange dual problem
        def loss(alpha):
             #'''--------------dual loss ------------------ '''
            return -alpha@np.ones(N) + 0.5 * (alpha*y) @ K @ (alpha*y)

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            # '''----------------partial derivative of the dual loss wrt alpha -----------------'''
            return np.diag(y) @ K @ (alpha*y) - np.ones(N)


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha: alpha @ y # '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda alpha:  y  #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha: np.hstack((np.zeros(N), self.C*np.ones(N))) + np.hstack((alpha, -alpha))  # '''---------------function defining the inequality constraint-------------------'''     
        jac_ineq = lambda alpha: np.vstack((np.eye(N), -1*np.eye(N))) # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac': jac_ineq})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x
        
        ## Assign the required attributes
        support_idxs  = np.argwhere(~np.isclose(self.alpha, 0)).squeeze()
        self.support = X[support_idxs]
        self.alpha_support = self.alpha[support_idxs]
        self.y_support = y[support_idxs]
        margin_idxs = np.intersect1d(support_idxs, np.argwhere(~np.isclose(self.alpha, self.C)).squeeze())
        self.margin_points = X[margin_idxs]#'''------------------- A matrix with each row corresponding to a point that falls on the margin ------------------'''
        
        self.b = np.mean(y[margin_idxs] - K[np.ix_(margin_idxs , support_idxs)] @ \
            (self.alpha_support*self.y_support))  #''' -----------------offset of the classifier------------------ '''
        self.norm_f = np.sqrt((self.alpha_support*self.y_support) @ \
            K[np.ix_(support_idxs , support_idxs)] @ (self.alpha_support*self.y_support))# '''------------------------RKHS norm of the function f ------------------------------'''


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x, self.support) @ (self.alpha_support*self.y_support)

    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1
    


class MulticlassKernelSVC:
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.models = []
        self.classes = []
    
    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in self.classes:
            y_bin = 2*(y == c)-1
            y_c_idxs = np.where(y_bin == 1)[0]
            num_c = len(y_c_idxs)
            y_nc_idxs = np.where(y_bin == -1)[0]
            # get the same number of samples for each class
            y_nc = np.random.choice(y_nc_idxs, num_c, replace=False)
            y_bin = y_bin[np.union1d(y_c_idxs, y_nc)]
            X_bin = X[np.union1d(y_c_idxs, y_nc)]
            model = KernelSVC(self.C, self.kernel, self.epsilon)
            model.fit(X_bin, y_bin)
            # model.fit(X, y_bin)
            self.models.append(model)
    
    def predict(self, X):
        pred = np.zeros((X.shape[0], len(self.classes)))
        for i, model in enumerate(self.models):
            pred[:, i] = model.separating_function(X) + model.b
        return self.classes[np.argmax(pred, axis=1)]