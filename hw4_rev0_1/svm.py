"""
An implementation of SVMs using cvxopt.

"""
import warnings
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


def kernel_dot(X1, X2, kernel_params):
    """
    Returns the elementwise kernel vector between X1 and X2.
    I.e. kernel_dot(X1, X2)_i = k(X1_i, X2_i)
    Parameters
    ----------
    X1 : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    X2 : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    np.ndarray (float64) of shape (n_samples,)
    """
    kp = kernel_params
    if kp['kernel'] == 'linear':
        return (X1 * X2).sum(1)
    elif kp['kernel'] == 'poly':
        return (kp['gamma'] * (X1 * X2).sum(1) + kp['coef0']) ** kp['degree']
    elif kp['kernel'] == 'rbf':
        return np.exp(-kp['gamma'] * ((X1 - X2)**2).sum(1))
    elif kp['kernel'] == 'sigmoid':
        return np.tanh(kp['gamma'] * (X1 * X2).sum(1) + kp['coef0'])
    else:
        raise ValueError(f"Unknown parameter: {kp['kernel']}")


def kernel_matrix(X1, X2, kernel_params):
    """
    Returns the pairwise kernel matrix between X1 and X2 (aka. gram matrix).
    I.e. kernel_dot(X1, X2)_{ij} = k(X1_i, X2_j)
    Parameters
    ----------
    X1 : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    X2 : np.ndarray (float64) of shape (m_samples, n_features)
        The input samples
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    np.ndarray (float64) of shape (n_samples, m_samples)
    """
    kp = kernel_params
    if kp['kernel'] == 'linear':
        return X1 @ X2.T
    elif kp['kernel'] == 'poly':
        return (kp['gamma'] * X1 @ X2.T + kp['coef0']) ** kp['degree']
    elif kp['kernel'] == 'rbf':
        pw_norm = ((np.expand_dims(X1, 1) - np.expand_dims(X2, 0))**2).sum(2)
        return np.exp(-kp['gamma'] * pw_norm)
    elif kp['kernel'] == 'sigmoid':
        return np.tanh(kp['gamma'] * X1 @ X2.T + kp['coef0'])
    else:
        raise ValueError(f"Unknown parameter: {kp['kernel']}")


def get_qp_params(X, y, C, kernel_params):
    """
    Return the parameters to pass into cvxopt.solvers.qp for the SVM dual problem.
    Parameters
    ----------
    X : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    y : np.ndarray (int64) of shape (n_samples,)
        Target labels, with values either -1 or 1.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    Arguments to be passed into cvxopt.solvers.qp
    P : ndarray of shape (n_samples, n_samples)
    q : ndarray of shape (n_samples,)
    G : ndarray of shape (n_1, n_samples)
    h : ndarray of shape (n_1,)
    A : ndarray of shape (n_2, n_samples)
    b : ndarray of shape (n_2,)
    """
    P, q, G, h, A, b = None, None, None, None, None, None

    ###################################################################################
    # TODO calculate the values for P, q, G, h, A, and b.
    # Make sure the shapes of these outputs are correct and pass the assertions below.
    # To calculate the kernel functions, refer to the instructions in the pdf or
    #   at https://scikit-learn.org/stable/modules/svm.html#svm-kernels
    # Note: you will not need all the parameters
    # (e.g. degree, gamma, coef0) for every kernel.
    # Hint: kernel_matrix may be useful here.
    ###################################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    n_samples, n_features = X.shape

    G = np.stack((np.diag( np.ones(n_samples) * -1), np.diag( np.ones(n_samples)) ) ).reshape( 2 * n_samples, n_samples )
    # print(G)
    
    h = np.stack( (np.zeros(n_samples), np.ones(n_samples) * C) ).reshape(-1)
    # print(h)
    
    P = ( (y.reshape(n_samples,-1) @ y.T.reshape(-1,n_samples)) * kernel_matrix(X, X, kernel_params) )

    q = np.ones(n_samples,) * -1

    A = y.reshape(-1,n_samples) * 1.
    
    b = np.zeros((A.shape[0],))
    # np.savetxt("temp.txt",b)
    ###################################################################################
    #                                END OF YOUR CODE                                 #
    ###################################################################################

    # Check for shapes
    assert P.dtype == q.dtype == G.dtype == h.dtype \
            == A.dtype == b.dtype == np.float_, 'outputs must be numpy floats'
    assert len(P.shape) == len(G.shape) == len(A.shape) == 2, 'P, G, and A must be matrices'
    assert len(q.shape) == len(h.shape) == len(b.shape) == 1, 'q, h, and b must be vectors'
    assert P.shape[0] == P.shape[1] == X.shape[0], 'wrong shape for P'
    assert q.shape[0] == X.shape[0], 'wrong shape for q'
    assert G.shape == (h.shape[0], X.shape[0]), 'wrong shape for G or h'
    assert A.shape == (b.shape[0], X.shape[0]), 'wrong shape for A or b'
    return P, q, G, h, A, b


def fit_bias(X, y, alpha, kernel_params):
    """
    Return the calculated values the bias b using the given support vector values (alpha).
    Parameters
    X : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples used to find alpha
    y : np.ndarray (int64) of shape (n_samples,)
        Target labels, with values either -1 or 1, used to find alpha
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -----
    Scalar (float64)
    """
    # Note: due to cvxopt qp implementation and numerical floats,
    #   alpha[i] may not be a support vector even if alpha[i] > 0
    #   Instead, we use:
    is_support = alpha > 1e-4
    ###################################################################################
    # TODO calculate the bias given X, y, and alpha
    # Hint: kernel_matrix or kernel_dot may be useful here
    ###################################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")

    X_support = X[is_support,:]
    Ns = np.sum(is_support)
    b = np.sum( y[is_support] -  np.dot(kernel_matrix(X_support, X_support, kernel_params), y[is_support]*alpha[is_support]) ) / Ns
    b = b.astype(np.float64)

    ###################################################################################
    #                                END OF YOUR CODE                                 #
    ###################################################################################
    return b


def decision_function(X, X_train, y_train, b, alpha, kernel_params):
    """
    Return the calculated values for (w^T X + b) using the given support vector values (alpha).
    Parameters
    ----------
    X : np.ndarray (float64) of shape (n_samples, n_features)
        The input samples
    X_train : np.ndarray (float64) of shape (n_train_samples, n_features)
        The input samples used to find alpha
    y_train : np.ndarray (int64) of shape (n_train_samples,)
        Target labels, with values either -1 or 1, used to find alpha
    b : scalar (float64)
        Bias value computed after training (along with alpha)
    alpha : np.ndarray (float64) of shape (n_train_samples,)
        Support vector values (solution of the cvxopt qp method)
    kernel_params : dictionary with parameters that specify the kernel
        parameters:
        kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
            default='rbf'
            Specifies the kernel type to be used in the algorithm.
        degree : int, default=3
            Degree of the polynomial kernel function ('poly').
            Must be non-negative. Ignored by all other kernels.
        gamma : float, default=1.0
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            - must be non-negative.
        coef0 : float, default=0.0
            Independent term in kernel function.
            It is only significant in 'poly' and 'sigmoid'.
    Outputs
    -------
    np.ndarray (float64) of shape (n_samples)
    """
    # Note: due to cvxopt qp implementation and numerical floats,
    #   alpha[i] may not be a support vector even if alpha[i] > 0
    #   Instead, we use:
    is_support = alpha > 1e-4
    h = np.zeros(X.shape[0])
    ###################################################################################
    # TODO calculate the values for h(x) = w^T X + b using the dual representation
    # Since this is kernelized, you may not be able to calculate w directly.
    # Hint: kernel_matrix or kernel_dot may be useful here
    ###################################################################################
    # raise NotImplementedError("TODO: Add your implementation here.")
    y_train_support = is_support * y_train
    alpha_support = is_support * alpha
    h = kernel_matrix(X,X_train,kernel_params) @ (alpha_support*y_train_support) + b
    # print(h.shape)
    # print(X.shape, X_train.shape)
    ###################################################################################
    #                                END OF YOUR CODE                                 #
    ###################################################################################
    assert h.shape == (X.shape[0],)
    return h


class CVXOPTSVC:
    """C-Support Vector Classification.

    Sklearn-style SVM implementation using the cvxopt solver.
    
    Parameters and interface are adapted from
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'},  \
        default='rbf'
        Specifies the kernel type to be used in the algorithm.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.
    gamma : float, default=1.0
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        - must be non-negative.
    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    """

    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma=1.0,
        coef0=0.0
    ):
        self.C = C
        self.kernel_params = {
            'kernel': kernel,
            'degree': degree,
            'gamma': gamma,
            'coef0': coef0
        }

    @staticmethod
    def _H_linear(X, y):
        Xy = X * np.expand_dims(y, 1)
        return Xy @ Xy.T

    def fit(self, X, y):
        # Initialize and computing H. Note the 1. to force to float type
        y = y * 2 - 1  # transform to [-1, 1]
        self.X_train = X.copy()
        self.y_train = y.copy()

        # Convert into cvxopt format
        _P, _q, _G, _h, _A, _b = get_qp_params(X, y, self.C, self.kernel_params)
        P = cvxopt_matrix(_P)
        q = cvxopt_matrix(np.expand_dims(_q, 1))
        G = cvxopt_matrix(_G)
        h = cvxopt_matrix(_h)
        A = cvxopt_matrix(_A)
        b = cvxopt_matrix(_b)

        # Run solver
        cvxopt_solvers.options['show_progress'] = False
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(sol['x']).squeeze(1)

        self.support_ = np.where(self.alpha > 1e-4)
        self.b = fit_bias(X, y, self.alpha, self.kernel_params)

        return self

    def decision_function(self, X):
        return decision_function(X, self.X_train, self.y_train, self.b, self.alpha, self.kernel_params)

    def predict(self, X):
        h = decision_function(X, self.X_train, self.y_train, self.b, self.alpha, self.kernel_params)
        return (h >= 0).astype(np.int_)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

if __name__ == "__main__":
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.model_selection import train_test_split

    # make linearly separable dataset
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    # make xor dataset
    # make xor dataset
    z = rng.binomial(1, 0.5, size=(100, 2)).astype(np.bool_)
    y = np.logical_xor(z[:, 0], z[:, 1]).astype(np.int_)
    X = rng.normal(loc=z, scale=0.2)
    xor_ds = (X, y)

    full_datasets = [
        make_moons(noise=0.3, random_state=0),
        make_circles(noise=0.2, factor=0.5, random_state=1),
        linearly_separable,
        xor_ds,
    ]

    datasets = []
    for ds in full_datasets:
        X, y = ds
        X = (X - X.mean(0)) / X.std(0)# normalize input
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        datasets.append({
            'X': X,
            'y': y,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        })
    
    from gradient_check import rel_error

    rng = np.random.default_rng(545)
    X = rng.normal(size=(5, 4))
    y = rng.normal(size=(5,))
    alpha = rng.normal(size=(5,))
    C = 1.0
    kernel_params = {'kernel': 'rbf', 'degree': 3, 'gamma': 0.2, 'coef0': 0.0}
    P, _, _, _, _, _ = get_qp_params(X, y, C, kernel_params)
    b = fit_bias(X, y, alpha, kernel_params)
    h = decision_function(X[:2], X, y, 545, alpha, kernel_params)

    P_sol = np.array([[ 0.45863879, -0.36760293, -0.09673795,  0.00553096, -0.01048321],
        [-0.36760293,  1.09673362,  0.06906934, -0.13144294,  0.13565593],
        [-0.09673795,  0.06906934,  0.46197662, -0.00232898,  0.01221352],
        [ 0.00553096, -0.13144294, -0.00232898,  1.84930408, -0.23840086],
        [-0.01048321,  0.13565593,  0.01221352, -0.23840086,  0.33240564]])

    b_sol = -0.5671911839511224
    h_sol = np.array([545.1811287 , 546.72502909])

    # Compare your output with ours. The error might be less than 1e-7.
    # As long as your error is small enough, your implementation should pass this test.
    print('Testing cvxopt functions:')
    print('difference: ', rel_error(P, P_sol))
    print('difference: ', rel_error(b, b_sol))
    print('difference: ', rel_error(h, h_sol))
    print()
    np.testing.assert_allclose(P, P_sol, atol=1e-6)
    np.testing.assert_allclose(b, b_sol, atol=1e-6)
    np.testing.assert_allclose(h, h_sol, atol=1e-6)



    kernels = ['linear', 'poly', 'sigmoid', 'rbf']
    cvxopt_models = [[CVXOPTSVC(kernel=k, gamma=0.2).fit(ds['X_train'], ds['y_train']) for k in kernels] for ds in datasets]
