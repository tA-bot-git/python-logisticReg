#from numpy import exp
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


def sigmoid(z):
    """
    Computes the sigmoid of z element-wise.

    Args:
        z: An np.array of arbitrary shape

    Returns:
        g: An np.array of the same shape as z

    """

    g = None
    #######################################################################
    # TODO:                                                               #
    # Compute and return the sigmoid of z in g                            #
    #######################################################################
    list = [] #Initialize a list to store the sigmoid values
    # for x in np.nditer(z): #iterate over the values of z
    #     summ = 1 / (1 + np.exp(-x)) #store the values of each sigmoid calculation of x in z
    #     list.append(summ) #append this sigmoid values to the list
    # g = np.array(list) #convert the list into a numpy array with all the computed sigmoid values
    g = 1 / (1 + np.exp(-z))
    pass

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return g
