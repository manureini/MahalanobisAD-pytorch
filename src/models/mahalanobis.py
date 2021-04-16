import numpy as np
import scipy
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import LedoitWolf

class MahalanobisAD:
    def __init__(self):
        self.threshold = scipy.stats.chi2.interval(0.99, 1)[1]

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.cov = LedoitWolf().fit(X).covariance_
        self.cov_inv = np.linalg.inv(self.cov)

    def score(self, X):
        score_array = np.zeros(len(X))
        for i in range(len(X)):
            score_array[i] = mahalanobis(X[i], self.mean, self.cov_inv)
        return score_array

