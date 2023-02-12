"""EECS545 HW1: Linear Regression."""

from typing import Any, Dict, Tuple

import numpy as np


def load_data():
    """Load the data required for Q2."""
    x_train = np.load('data/q2xTrain.npy')
    y_train = np.load('data/q2yTrain.npy')
    x_test = np.load('data/q2xTest.npy')
    y_test = np.load('data/q2yTest.npy')
    return x_train, y_train, x_test, y_test


def generate_polynomial_features(x: np.ndarray, M: int) -> np.ndarray:
    """Generate the polynomial features.

    Args:
        x: A numpy array with shape (N, ).
        M: the degree of the polynomial.
    Returns:
        phi: A feature vector represented by a numpy array with shape (N, M+1);
          each row being (x^{(i)})^j, for 0 <= j <= M.
    """
    N = len(x)
    phi = np.zeros((N, M + 1))
    for m in range(M + 1):
        phi[:, m] = np.power(x, m)
    return phi


def loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    r"""The least squares training objective for the linear regression.

    Args:
        X: the feature matrix, with shape (N, M+1).
        y: the target label for regression, with shape (N, ).
        w: the linear regression coefficient, with shape (M+1, ).
    Returns:
        The least square error term with respect to the coefficient weight w,
        E(\mathbf{w}).
    """
    y_pred = np.matmul(X, w)
    squared_error = (y - y_pred) ** 2
    return 0.5 * np.sum(squared_error)


def MSE(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Returns mean squared error for the linear regression.

    Args:
        X: the feature matrix, with shape (N, M+1).
        y: the target label for regression, with shape (N, ).
        w: the linear regression coefficient, with shape (M+1, ).
    Returns:
        The mean squared error with respect to the coefficient weight w.
    """
    y_pred = np.matmul(X, w)
    squared_error = (y - y_pred) ** 2
    mse = np.mean(squared_error)
    return mse


def batch_gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    eta: float = 0.01,
    max_epochs: int = 10000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Batch gradient descent for linear regression that fits the
    feature matrix `X_train` to target `y_train`.

    Args:
        X_train: the feature matrix, with shape (N, M+1).
        y_train: the target label for regression, with shape (N, ).
        eta: Learning rate.
        max_epochs: Maximum iterations (epochs) allowed.
    Returns: A tuple (w, info)
        w: The coefficient of linear regression found by GD. Shape (M+1, ).
        info: A dict that contains additional information (see the notebook).
    """
    ###################################################################
    # TODO: Implement the Batch GD solver.
    ###################################################################
    M = X_train.shape[1]-1
    # print(M)
    w = np.zeros((M+1,))
    info = {}
    info["train_losses"] = []
    for n in range(max_epochs):
        temp = np.matmul(np.transpose(X_train),X_train)
        grad_w = np.matmul(temp,w) - np.matmul(np.transpose(X_train),y_train)
        w = w - eta * grad_w
        loss_gd = loss(X_train, y_train, w)
        info["train_losses"].append(loss_gd)
        if loss_gd <= 0.2 * y_train.shape[0]:
            break
    info["number of epochs"] = n
    # print(info)
    # raise NotImplementedError("TODO: Add your implementation here.")
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    return w, info


def stochastic_gradient_descent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    eta=4e-2,
    max_epochs=10000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stochastic gradient descent for linear regression that fits the
    feature matrix `X_train` to target `y_train`.

    Args:
        X_train: the feature matrix, with shape (N, M+1).
        y_train: the target label for regression, with shape (N, ).
        eta: Learning rate.
        max_epochs: Maximum iterations (epochs) allowed.
    Returns: A tuple (w, info)
        w: The coefficient of linear regression found by SGD. Shape (M+1, ).
        info: A dict that contains additional information (see the notebook).
    """
    ###################################################################
    # TODO: Implement the SGD solver.
    ###################################################################
    M = X_train.shape[1]-1
    w = np.zeros((M+1,))
    info = {}
    info["train_losses"] = []
    for t in range(max_epochs):
        for n in range(X_train.shape[0]):
            for i in range(M+1):
                w[i] = w[i] - eta * (np.matmul(w.reshape(1,2),X_train[n].reshape(2,1))-y_train[n]) * X_train[n,i]
        loss_gd = loss(X_train, y_train, w)
        info["train_losses"].append(loss_gd)
        if loss_gd <= 0.2 * y_train.shape[0]:
            break

    info["number of epochs"] = t
    # raise NotImplementedError("TODO: Add your implementation here.")
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    return w, info
    # for t in range(max_epochs):
    #     for n in range(X_train.shape[0]):
    #         phi_n = X_train[n,:].reshape(1,M+1)
    #         temp = np.matmul(np.transpose(phi_n),phi_n)
    #         grad_w = np.matmul(temp,w.reshape(2,1))- np.matmul(np.transpose(phi_n),y_train[n].reshape(1,1))
    #         if ( (np.maximum(grad_w,-grad_w) < 1e-12).all() ):
    #             break
    #         w = w.reshape(2,1) - eta * grad_w
    #     if ( (np.maximum(grad_w,-grad_w) < 1e-12).all() ):
    #             break

def closed_form(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    lam: float = 0.0,
) -> np.ndarray:
    """Return the closed form solution of linear regression.

    Arguments:
        X_train: The X feature matrix, shape (N, M+1).
        y_train: The y vector, shape (N).
        M: The degree of the polynomial to generate features for.
        lam: The regularization coefficient lambda.

    Returns:
        The (optimal) coefficient w for the linear regression problem found,
        a numpy array of shape (M+1, ).
    """
    ###################################################################
    # TODO: Implement the closed form solution.
    ###################################################################
    M = X_train.shape[1] - 1
    w_ml = np.zeros((M+1))
    temp = np.linalg.inv( lam*np.identity(M+1) + np.matmul(np.transpose(X_train),X_train))
    w_ml = np.matmul(temp, np.matmul(np.transpose(X_train), y_train))
    # print(loss(X_train, y_train, w_ml))
    w = w_ml
    
    # raise NotImplementedError("TODO: Add your implementation here.")
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    return w


def closed_form_locally_weighted(
    X_train: np.ndarray,
    y_train: np.ndarray,
    r_train: np.ndarray,
) -> np.ndarray:
    """Return the closed form solution of locally weighted linear regression.

    Arguments:
        x_train: The X feature matrix, shape (N, M+1).
        y_train: The y vector, shape (N, ).
        r_train: The local weights for data point. Shape (N, ).

    Returns:
        The (optimal) coefficient for the locally weighted linear regression
        problem found. A numpy array of shape (M+1, ).
    """
    
    ###################################################################
    # TODO: Implement the closed form solution.
    ###################################################################
    M = X_train.shape[1]
    X_train_trans = np.transpose(X_train)
    R_train = np.diag(r_train)
    
    temp = np.matmul(X_train_trans,R_train)
    temp = np.matmul(temp, X_train)
    temp = np.linalg.inv(temp)
    temp = np.matmul(temp, X_train_trans)
    temp = np.matmul(temp, R_train)
    w = np.matmul(temp, y_train)

    # raise NotImplementedError("TODO: Add your implementation here.")
    ###################################################################
    #                        END OF YOUR CODE                         #
    ###################################################################
    return w
