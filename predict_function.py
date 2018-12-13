from sigmoid import sigmoid
import numpy as np


def predict_function(theta, X, y=None):
    """
    Compute predictions on X using the parameters theta. If y is provided
    computes and returns the accuracy of the classifier as well.
    """

    preds = None
    accuracy = None    
    threshold = 0.5
    score = np.dot(X, theta)
    preds_1 = sigmoid(score)
    preds = np.where(preds_1 >=threshold, 1, 0)    
    accuracy = np.mean(y == preds)        
    
    return preds, accuracy
