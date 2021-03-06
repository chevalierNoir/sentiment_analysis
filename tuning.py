from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import *

import numpy as np
import cvxopt

cvxopt.solvers.options['show_progress'] = False

#%matplotlib inline

class Kernel(object):
    """
    A class containing all kinds of kernels.
    Note: the kernel should work for both input (Matrix, vector) and (vector, vector)
    """
    @staticmethod
    def linear():
        def f(x, y):
            return np.dot(x, y)
        return f

    @staticmethod
    def gaussian(gamma):  # we use the commonly used name, although it's not really a Gaussian
        def f(x, y):
            exponent = - gamma * np.linalg.norm((x-y).transpose(), 2, 0) ** 2
            return np.exp(exponent)
        return f

    @staticmethod
    def _poly(dimension, offset):
        def f(x, y):
            return (offset + np.dot(x, y)) ** dimension
        return f

    @staticmethod
    def inhomogenous_polynomial(dimension):
        return Kernel._poly(dimension=dimension, offset=1.0)

    @staticmethod
    def homogenous_polynomial(dimension):
        return Kernel._poly(dimension=dimension, offset=0.0)

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        def f(x, y):
            return np.tanh(kappa * np.dot(x, y) + c)
        return f

class SVM(object):
    def __init__(self, kernel, c):
        """
        Build a SVM given kernel function and C

        Parameters
        ----------
        kernel : function
            a function takes input (Matrix, vector) or (vector, vector)
        c : a scalar
            balance term

        Returns
        -------
        """
        self._kernel = kernel
        self._c = c

    def fit(self, X, y):
        """
        Fit the model given data X and ground truth label y

        Parameters
        ----------
        X : 2D array
            N x d data matrix (row per example)
        y : 1D array
            class label

        Returns
        -------
        """
        # Solve the QP problem to get the multipliers
        lagrange_multipliers = self._compute_multipliers(X, y)
        # Get all the support vectors, support weights and bias
        self._construct_predictor(X, y, lagrange_multipliers)
    
    def predict(self, X):
        """
        Predict the label given data X

        Parameters
        ----------
        X : 2D array
            N x d data matrix (row per example)

        Returns
        -------
        y : 1D array
            predicted label
        """
        result = np.full(X.shape[0], self._bias) # allocate
        
#         YOUR CODE HERE
        for ind in range(X.shape[0]):
            K = self._kernel(self._support_vectors, X[ind])
            result[ind] = np.sum(np.multiply(self._weights, np.multiply(self._support_vector_labels, K))) + self._bias
        return np.sign(result)

    def _kernel_matrix(self, X):
        """
        Get the kernel matrix.

        Parameters
        ----------
        X : 2D array
            N x d data matrix (row per example)

        Returns
        -------
        K : 2D array
            N x N kernel matrix
        """
        N, d = X.shape
        K = np.zeros((N, N))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):
        """
        Given the data, label and the multipliers, extract the support vectors and calculate the bias

        Parameters
        ----------
        X : 2D array
            N x d data matrix (row per example)
        y : 1D array
            class label
        lagrange_multipliers: 1D array
            the solution of lagrange_multiplier

        Fills in relevant variables: model bias and weights (alphas), and details of support vectors
        
        -------
        """
        support_vector_indices = \
            lagrange_multipliers > 1e-5
            
        print("SV number: ", np.sum(support_vector_indices))

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        """
        Get the bias term (w_0)
        """
#         YOUR CODE HERE
        # alpha<C and y = +1 or -1
#         indm = []
#         indp = []
        K = self._kernel_matrix(X)
#         for index in support_vector_indices:
#             if y[index] == 1 and support_multipliers[index] < self._c - 1e-5:
#                 indp.append(index)
#             if y[index] == 0 and support_multipliers[index] < self._c - 1e-5:
#                 indm.append(index)
#         print("indm:", indm)
#         print("indp:", indp)

        support_kernels = K.T[support_vector_indices]
        sv_number = np.sum(support_vector_indices)
        N = y.shape
#         km = np.zeros(N)
#         kp = np.zeros(N)
#         nkm = 0
#         nkp = 0
#         for ind in range(sv_number):
#             if support_multipliers[ind] < self._c - 1e-5 and y[ind] == 1:
#                 nkp = nkp + 1.0
#                 kp = (kp + support_kernels[ind])/nkp
#             if support_multipliers[ind] < self._c - 1e-5 and y[ind] == -1:
#                 nkm = nkm + 1.0
#                 km = (km + support_kernels[ind])/nkm      
#         bm = np.sum(np.multiply(lagrange_multipliers, np.multiply(y, km)))
#         bp = np.sum(np.multiply(lagrange_multipliers, np.multiply(y, kp)))
#         bias = -(bm + bp)/2.0
        omega0 = []
        for ind in range(sv_number):
            if support_multipliers[ind] < self._c - 1e-5:
                omega0.append(1.0/y[ind] - np.sum(np.multiply(lagrange_multipliers, np.multiply(y, support_kernels[ind]))))
        bias = np.mean(omega0)
        print("bias mean/std:{}/{}".format(bias, np.std(omega0)))
    
        self._bias=bias
        self._weights=support_multipliers
        self._support_vectors=support_vectors
        self._support_vector_labels=support_vector_labels

    def _compute_multipliers(self, X, y):
        """
        Given the data, label, solve the QP program to get lagrange multiplier.

        Parameters
        ----------
        X : 2D array
            N x d data matrix (row per example)
        y : 1D array
            class label

        Returns
        lagrange_multipliers: 1D array
        -------
        """
        N, d = X.shape

        K = self._kernel_matrix(X)
        """
        The standard QP solver formulation:
        min 1/2 x^T H x + f^T x
        s.t.
        Ax <=  a
        Bx = b
        """
        H = np.dot(np.dot(np.diag(y), K),np.diag(y))
        H = cvxopt.matrix(H)
        f = -np.ones([N])
        f = cvxopt.matrix(f)

        A = np.concatenate((np.eye(N), -np.eye(N)), axis = 0)
        a = np.concatenate((self._c * np.ones(N), np.zeros(N)))
        A = cvxopt.matrix(A)
        a = cvxopt.matrix(a)
        
        

        B = y
        b = np.zeros([1])
        B = cvxopt.matrix(B).T
        b = cvxopt.matrix(b)
#         print(B.size)
        
        # call the QP solver
        solution = cvxopt.solvers.qp(H, f, A, a, B, b)

        # Lagrange multipliers (the unknown vector 'x' is our alphas)
        return np.ravel(solution['x'])


bigram_range = [True, False]
mincount_range = range(1, 10)
c_range = [pow(10, i) for i in range(-2, 3)]

c_max = 0
acc_max = -10
mincount_max = 0
bigram_max = False

print("#############Linear Classifier##############")
for bigram in bigram_range:
    for mincount_ in mincount_range:
        X, y, keys = preprocess(use_bigram = bigram, mincount = mincount_)
        c_max = 0
        acc_max = -10
        for C in c_range:
            clf = SVM(Kernel.linear(), C)
            clf.fit(X['train'], y['train'].astype('double'))
            print("C = ", C)
            y_hat = clf.predict(X['train'])
            print("Acc on train: ", np.mean(y_hat == y['train']))
            y_hat = clf.predict(X['val'])
            print("Acc on val: ", np.mean(y_hat == y['val']))
            acc = np.mean(y_hat == y['val'])
            if acc > acc_max:
                c_max = C
                acc_max = acc
                bigram_max = bigram
                mincount_max = mincount_
print("best c: {}, mincount: {}, bigram: {}, max acc: {}".format(c_max, mincount_max, bigram_max, acc_max))

'''
print("##############Gaussian Kernel######################")
c_range = [1.0, 10.0, 100.0, 1000.0]
gamma_range = [0.01, 0.1, 1]
c_max = 0
gamma_max = 0
acc_max = -10
for bigram in bigram_range:
    for mincount_ in mincount_range:
        X, y, keys = preprocess(use_bigram = bigram, mincount = mincount_)
        for C in c_range:
            for gamma in gamma_range:
                clf = SVM(Kernel.gaussian(gamma), C)
                clf.fit(X['train'], y['train'].astype('double'))
                print("C = ", C)
                print("gamma = ", gamma)
                y_hat = clf.predict(X['train'])
                print("Acc on train: ", np.mean(y_hat == y['train']))
                y_hat = clf.predict(X['val'])
                print("Acc on val: ", np.mean(y_hat == y['val']))
                acc = np.mean(y_hat == y['val'])
                if acc > acc_max:
                    gamma_max = gamma
                    c_max = C
                    bigram_max = bigram
                    mincount_max = mincount_
                    acc_max = acc
print("best c: {}, best gamma: {}, best bigram: {}, best mincount: {}, max acc: {}".format(c_max, gamma_max, bigram_max, mincount_max, acc_max))
'''     
