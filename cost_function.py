from sigmoid import sigmoid
import math
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
   
    i = len(y[np.where(y==0)])    
    j = len(y[np.where(y==1)])        
    
    m,n = X.shape
    sigm = sigmoid(np.dot(theta.T, X.T)) #fine    
    h_x = np.sum(sigm)/m     
    a = np.log10(1 - h_x)           
    var1 = i * a    
    var2 = j  * np.log10(h_x)        
    
    l  = np.sum((var1 + var2))
    
    return l
