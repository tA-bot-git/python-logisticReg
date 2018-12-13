from cost_function import cost_function
from gradient_function import gradient_function
from sigmoid import sigmoid
import numpy as np
import time


def logistic_SGD(X, y, num_iter=10000, alpha=0.01):
    """
    Perform logistic regression with stochastic gradient descent.

    Args:
        theta_0: Initial value for parameters of shape [num_features]
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train, 1]
        num_iter: Number of iterations of SGD
        alpha: The learning rate

    Returns:
        theta: The value of the parameters after logistic regression

    """

    theta = np.zeros(X.shape[1])
    losses = []
    new_loss = cost_function(theta,X,y)
    for i in range(num_iter):
        start = time.time()
        
        N = len(X)
        #
        theta_transp = np.transpose(theta)
        theta_x = np.dot(X, theta_transp)
        predictions = sigmoid(theta_x)
        #
        #grad = gradient_function(theta, X, y)
        gradient = np.dot(X.T, predictions - y)
        #
        gradient /= N
        #
        gradient *= alpha
        #
        theta -= gradient
        #
        # return theta

        if i % 1000 == 0:
            exec_time = time.time() - start
            loss = cost_function(theta, X, y)
            losses.append(loss)
            print('Iter {}/{}: cost = {}  ({}s)'.format(i, num_iter, loss, exec_time))
            alpha *= 0.9

    return theta, losses
