import numpy as np
from scipy import optimize
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

class KernelSVC:
    def __init__(self, C, kernel):
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.norm_f = None
       
    
    def fit(self, X, y):
        N = len(y)
        K = self.kernel(X,X)
        
        def loss(alpha):
            return -alpha@np.ones(N) + 0.5 * (alpha*y) @ K @ (alpha*y)

        def grad_loss(alpha):
            return np.diag(y) @ K @ (alpha*y) - np.ones(N)

        fun_eq = lambda alpha: alpha @ y #equality constraint
        jac_eq = lambda alpha:  y  #jacobian wrt alpha of the equality constraint
        fun_ineq = lambda alpha: np.hstack((np.zeros(N), self.C*np.ones(N))) + np.hstack((alpha, -alpha))  #inequality constraint
        jac_ineq = lambda alpha: np.vstack((np.eye(N), -1*np.eye(N))) #jacobian wrt alpha of the inequality constraint
        
        constraints = (
            {'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac': jac_ineq})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x
        
        support_idxs  = np.argwhere(~np.isclose(self.alpha, 0)).squeeze()
        self.support = X[support_idxs]
        self.alpha_support = self.alpha[support_idxs]
        self.y_support = y[support_idxs]
        margin_idxs = np.intersect1d(support_idxs, np.argwhere(~np.isclose(self.alpha, self.C)).squeeze())
        self.margin_points = X[margin_idxs]
        
        self.b = np.mean(y[margin_idxs] - K[np.ix_(margin_idxs , support_idxs)] @ \
            (self.alpha_support*self.y_support))
        self.norm_f = np.sqrt((self.alpha_support*self.y_support) @ \
            K[np.ix_(support_idxs , support_idxs)] @ (self.alpha_support*self.y_support))


    def separating_function(self,x):
        return self.kernel(x, self.support) @ (self.alpha_support*self.y_support)

    
    
    def predict(self, X):
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1


class MulticlassKernelSVC:
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.models = []
        self.classes = []
    
    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in tqdm(self.classes):
            y_bin = 2*(y == c)-1
            y_c_idxs = np.where(y_bin == 1)[0]
            num_c = len(y_c_idxs)
            y_nc_idxs = np.where(y_bin == -1)[0]
            y_nc = np.random.choice(y_nc_idxs, num_c, replace=False)
            y_bin = y_bin[np.union1d(y_c_idxs, y_nc)]
            X_bin = X[np.union1d(y_c_idxs, y_nc)]
            model = KernelSVC(self.C, self.kernel)
            model.fit(X_bin, y_bin)
            self.models.append(model)
    
    def predict(self, X):
        pred = np.zeros((X.shape[0], len(self.classes)))
        for i, model in tqdm(enumerate(self.models)):
            pred[:, i] = model.separating_function(X) + model.b
        return self.classes[np.argmax(pred, axis=1)]

class OneVsOneKernelSVC:
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.models = []
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for i, c1 in tqdm(enumerate(self.classes)):
            self.models.append([])
            for j, c2 in enumerate(self.classes[i+1:]):
                y_bin_idxs = np.where((y == c1) | (y == c2))[0]
                y_bin = 2*(y[y_bin_idxs] == c1)-1
                X_bin = X[(y == c1) | (y == c2)]
                model = KernelSVC(self.C, self.kernel)
                model.fit(X_bin, y_bin)
                self.models[i].append((c1,c2,model))
    
    def predict(self, X):
        pred = np.zeros((X.shape[0], len(self.classes)))
        for i, models in tqdm(enumerate(self.models)):
            for c1,c2,model in models:
                pred_ = model.predict(X)
                for k, p in enumerate(pred_):
                    if p == 1:
                        pred[k,c1] += 1
                    else:
                        pred[k,c2] += 1
        return self.classes[np.argmax(pred, axis=1)]

    
class MultivariateKernelRidgeClassifier:      
    def __init__(self,kernel,lmbda):
        self.lmbda = lmbda                    
        self.kernel = kernel
        self.support = None
        self.alpha = None
        self.b = None
    
    def fit(self, X, y):
        self.support = X
        Y = LabelBinarizer().fit_transform(y)
        self.alpha = np.zeros((X.shape[0],Y.shape[1]))
        self.b = np.zeros(Y.shape[1])
        K = self.kernel(X,X)
        for i in tqdm(range(Y.shape[1])):
            self.alpha[:,i] = np.linalg.solve(K + self.lmbda*K.shape[0]*np.eye(K.shape[0]) \
                + np.ones(K.shape[0]).reshape(-1,1)@np.mean(K, axis=0).reshape(1,-1), Y[:,i] - \
                    np.mean(Y[:,i]))    
        
            self.b[i] = np.mean(Y[:,i]) - np.mean(K @ self.alpha[:,i])

        
    def regression_function(self,x):
        K = self.kernel(x,self.support)  
        return K @ self.alpha
    
    def predict(self, X):
        return np.argmax(self.regression_function(X)+np.expand_dims(self.b,axis=0), axis=1)