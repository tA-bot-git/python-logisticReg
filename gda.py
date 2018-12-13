from cost_function import cost_function
import numpy as np
import time


def gda(X, y):
    """
    Perform Gaussian Discriminant Analysis.

    Args:
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train, 1]

    Returns:
        theta: The value of the parameters after logistic regression
    """
    # Initialize Variables
    theta = None
    phi = None
    mu_0 = None
    mu_1 = None
    sigma = None
           
    X = X[:, 1:]    # Note: Remove the bias term!
    start = time.time()
    
    m,n  = X.shape       
    phi  = np.sum(np.where(y == 1,1,0), dtype=float)/m 
                                   
    a = tuple(np.array(np.where(y == 0)).tolist())
    b = tuple(np.array(np.where(y == 1)).tolist())
    
    mu_0 = np.sum(X[tuple(a)],axis=0, dtype = float)/len(y[a]) 
    mu_1 = np.sum(X[tuple(b)],axis=0, dtype = float)/len(y[b])        
            
    var0 = np.array(X[a] - mu_0)/len(a)  
    var1 = np.array(X[b] - mu_1)/len(b)              
           
    co_variances = np.concatenate((var0, var1), axis=0)/m
         
    sigma = np.dot(co_variances.T, co_variances)        

    # Compute theta from the results of GDA
    sigma_inv = np.linalg.inv(sigma)
    quad_form = lambda A, x: np.dot(x.T, np.dot(A, x))
    b = 0.5*quad_form(sigma_inv, mu_0) - 0.5*quad_form(sigma_inv, mu_1) + np.log(phi/(1-phi))
    w = np.dot((mu_1-mu_0), sigma_inv)
    theta = np.concatenate([[b], w])
    exec_time = time.time() - start

    # Add the bias to X and compute the cost
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    loss = cost_function(theta, X, y)

    print('Iter 1/1: cost = {}  ({}s)'.format(loss, exec_time))

    return theta, None
