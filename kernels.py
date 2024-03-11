import numpy as np

class LinearKernel:
    def __call__(self, X1, X2):
        return X1 @ X2.T
class PolynomialKernel:
    def __init__(self, p=3):
        self.p = p
    def __call__(self, X1, X2):
        return (X1 @ X2.T + 1) ** self.p

class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def __call__(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        dists = np.square(X)[:, np.newaxis].sum(axis=2) - 2*X @ Y.T + np.square(Y).sum(axis=1)
        return np.exp(-dists/(2*self.sigma**2))