from numpy.core.tests.test_mem_overlap import xrange
from cost_function import  cost_function
from sigmoid import sigmoid
import numpy as np


def gradient_function(theta, X, y):
    """
    Compute gradient for logistic regression w.r.t. to the parameters theta.

    Args:
        theta: Parameters of shape [num_features]
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]

    Returns:
        grad: The gradient of the log-likelihood w.r.t. theta

    """

    grad = None
    
    theta_transp = np.transpose(theta)
    X_transp = np.transpose(X)
    pi_i = np.dot(X, theta_transp)
    sigm  = sigmoid(pi_i)
    hyp = y-sigm
    grad = np.dot(X_transp, hyp)
    
    return grad
