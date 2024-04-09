import numpy as np
from tqdm import tqdm

class LinearKernel:
    def __init__(self):
        pass
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        return X @ Y.T

class PolynomialKernel:
    def __init__(self, p=3):
        self.p = p
    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        return (X1 @ X2.T + 1) ** self.p

class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma
    def __call__(self,X,Y=None):
        if Y is None:
            Y = X
        dists = np.square(X)[:, np.newaxis].sum(axis=2) - 2*X @ Y.T + np.square(Y).sum(axis=1)
        return np.exp(-dists/(2*self.sigma**2))

class LaplacianKernel():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        n = X.shape[0]
        m = Y.shape[0]
        K = np.zeros((n, m))

        for i in tqdm(range(m)):
            K[:, i] = np.sum(np.abs(X - Y[i, :]), axis=1)
        K /= self.sigma ** 2
        return np.exp(-K)