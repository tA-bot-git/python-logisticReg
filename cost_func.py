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
    
    observations = len(y)

    theta_transp = np.transpose(theta)
    theta_x = np.dot(X, theta_transp)
    predictions = sigmoid(theta_x)

    class1_cost = -y * np.log(predictions)

    class2_cost = (1 - y) * np.log(1 - predictions)

    cost = class1_cost - class2_cost

    l = np.sum(cost) / observations
    pass
    pass

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return l
