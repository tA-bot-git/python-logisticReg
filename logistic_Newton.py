from cost_function import cost_function
from gradient_function import gradient_function
from sigmoid import sigmoid
import numpy as np
import time


def logistic_Newton(X, y, num_iter=10):
    """
    Perform logistic regression with Newton's method.

    Args:
        theta_0: Initial value for parameters of shape [num_features]
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train, 1]
        num_iter: Number of iterations of Newton's method

    Returns:
        theta: The value of the parameters after logistic regression

    """

    theta = np.zeros(X.shape[1])
    losses = []
    for i in range(num_iter):
        start = time.time()
        #######################################################################
        # TODO:                                                               #
        # Perform one step of Newton's method:                                #
        #   - Compute the Hessian                                             #
        #   - Update theta using the gradient and the inverse of the hessian  #
        #                                                                     #
        # Hint: To solve for A^(-1)b consider using np.linalg.solve for speed #
        #######################################################################
        # Computing the Hessian
        theta_transp = np.transpose(theta)
        x_transp = np.transpose(X)
        score = np.dot(X, theta_transp)
        mu = 1 / 1 + np.exp(score)
        scores = np.dot(np.transpose(mu), (1 - mu))  # line represents: mu multiplied by 1 minus mu
        X_xtransp = np.dot(x_transp, X)  # x multiplied by x transposed
        #Hessian = np.sum(np.dot(X_xtransp, scores))
        Hessian = np.dot(X_xtransp, scores)  
        inverse_ = np.linalg.inv(Hessian)    


        # Update theta using gradient and inverse of Hessian
        grad = gradient_function(theta, X, y)
        #theta = theta - np.linalg.solve(inverse_, grad)
        var = np.linalg.solve(inverse_, grad)
        var_transp = np.transpose(var)
        theta = theta - np.dot(grad,var_transp)
        
        pass

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        exec_time = time.time()-start
        loss = cost_function(theta, X, y)
        losses.append(loss)
        print('Iter {}/{}: cost = {}  ({}s)'.format(i+1, num_iter, loss, exec_time))

    return theta, losses
