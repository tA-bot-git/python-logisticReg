from numpy import exp
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
    g = 1 / (1 + np.exp(-z))    
    
    return g
