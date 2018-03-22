from sigmoid import sigmoid
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def cost_function(theta, X, y):
    """
    Computes the cost of using theta as the parameter for logistic regression

    Args:
        theta: Parameters of shape [num_features]
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]

    Returns:
        l: The cost for logistic regression

    """

    l = None
    #######################################################################
    # TODO:                                                               #
    # Compute and return the log-likelihood l of a particular choice of   #
    # theta.                                                              #
    #                                                                     #
    #######################################################################
    theta_transp = np.transpose(theta)
    theta_x = np.multiply(theta_transp , X)
    hyp = sigmoid(theta_x)
    var1 = np.log(hyp)
    var1_transp= np.transpose(var1)
    var2 = np.multiply(var1_transp , y)
    var3 = 1 - y
    err = 1 - hyp
    var4 = np.log(err)
    var4_transp = np.transpose(var4)
    var5 = np.multiply(var4_transp , var3)

    l  = np.sum(-(var2 + var5))/X.__len__()

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return l
