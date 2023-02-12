"""EECS545 HW2: Logistic Regression."""

import numpy as np
import math


def hello():
    print('Hello from logistic_regression.py')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def naive_logistic_regression(X: np.ndarray, Y: np.ndarray, max_iters = 100) -> np.ndarray:
    """Computes the coefficients w from the datset (X, Y).

    This implementation uses a naive set of nested loops over the data.
    Specifically, we are required to use Newton's method (w = w - inv(H)*grad).

    Inputs:
      - X: Numpy array of shape (num_data, num_features+1).
           The first column of each row is always 1.
      - Y: Numpy array of shape (num_data) that has 0/1.
      - max_iters: Maximum number of iterations
    Returns:
      - w: Numpy array of shape (num_features+1) w[i] is the coefficient for the i-th
           column of X. The dimension should be matched with the second dimension of X.
    """
    N, d = X.shape
    w = np.zeros(d, dtype=X.dtype)
    for iter in range(max_iters):
        grad = np.zeros(d)
        H = np.zeros((d, d))
        for data_x, data_y in zip(X, Y):
            ###################################################################
            # TODO: Implement this function using Newton's method.            #
            # Specifically, you are required to update the following two      #
            # variables described below.                                      #
            # * grad: accumulated gradient over data samples.                 #
            # * H: accumulated Hessian matrix over data samples.              #
            # Please do not check a convergence condition for this time.      #
            # Also, please do not add any additional python inner loop here.  #
            # note: You are allowed to use predefined sigmoid function above. #
            ###################################################################
            # raise NotImplementedError("TODO: Add your implementation here.")
            sig = sigmoid(np.matmul(w.reshape(1,-1), data_x.reshape(-1,1)).reshape(1))
            H = H - np.matmul(data_x.reshape(-1,1),data_x.reshape(1,-1)) * sig * (1-sig)
          
            grad = grad + (data_y - sigmoid( np.matmul(w.reshape(1,-1), data_x.reshape(-1,1)).reshape(1) )) * data_x
            ###################################################################
            #                        END OF YOUR CODE                         #
            ###################################################################
        w = w - np.matmul(np.linalg.inv(H), grad)
    return w


def vectorized_logistic_regression(X: np.ndarray, Y: np.ndarray, max_iters = 100) -> np.ndarray:
    """Computes the coefficients w from the dataset (X, Y).

    This implementation will vectorize the implementation in naive_logistic_regression,
    which implements Newton's method (w = w - inv(H)*grad).

    Inputs:
      - X: Numpy array of shape (num_data, num_features+1).
           The first column of each row is always 1.
      - Y: Numpy array of shape (num_data) that has 0/1.
      - max_iters: Maximum number of iterations
    Returns:
      - w: Numpy array of shape (num_features+1) w[i] is the coefficient for the i-th
           column of X. The dimension should be matched with the second dimension of X.
  """
    N, d = X.shape
    w = np.zeros(d, dtype=X.dtype)
    for iter in range(max_iters):
        #######################################################################
        # TODO: Implement this function using Newton's method.                #
        # Because this version does not have any inner loop, compared to the  #
        # previous (naive_logistic_regression) task, you are required to      #
        # compute grad and H at once.                                         #
        # * grad: gradient from all data samples.                             #
        # * H: Hessian matrix from all data samples.                          #
        # The shape of grad and H is the same as 'naive_logistic_regression'. #
        # Please do not check a convergence condition for this time.          #
        # Also, please do not add any additional python inner loop here.      #
        # You are allowed to use predefined sigmoid function above.           #
        #######################################################################
        grad = None  # hint: grad.shape should be (d, ) at the end of this block
        H = None  # hint: H.shape should be (d, d) at the end of this block
        # raise NotImplementedError("TODO: Add your implementation here.")
        # print(np.matmul(X, w).shape)
        sig = sigmoid(np.matmul(X, w).reshape(-1))
        A = np.diag(sig * (1-sig))
        H = - np.matmul(np.matmul(X.T, A) ,X)
        grad = np.matmul(X.T , Y - sigmoid(np.matmul(X , w))).reshape(-1)
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        w = w - np.matmul(np.linalg.inv(H), grad)
    return w


def compute_y_boundary(X_coord: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the matched y coordinate value for the decision boundary from
    the x coordinate and coefficients w.

    Inputs:
      - X_coord: Numpy array of shape (d, ). List of x coordinate values.
      - w: Numpy array of shape (3, ) that stores the coefficients.

    Returns:
      - Y_coord: Numpy array of shape (d, ).
                 List of y coordinate values with respect to the coefficients w.
    """
    Y_coord = None
    ###########################################################################
    # TODO: Compute y_coordinate of the decision boundary with respect to     #
    # x_coord and coefficients w. Please return/save your y_coordindate into  #
    # y_coord parameter. It is fair to assume that w[2] will not be zero.     #
    ###########################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    
    # h(x) = sigmoid(w.T * x(n) ) = 0.5
    # w.T * x(n) = 0
    # w0 * 1 + w1 * x1 + w2 * x2 = 0

    Y_coord =  (-w[0] - w[1] * X_coord) / w[2] 

    # print( w )
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

    return Y_coord
