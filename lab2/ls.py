import numpy as np
import pickle

class LeastSquares(object):
    def __init__(self, k):
        """
        Initialize the LeastSquares class

        The input parameter k specifies the degree of the polynomial
        """
        self.k = k
        self.coeff = None

    def fit(self, x, y):
        """
        Find coefficients of polynomial that predicts y given x with
        degree self.k

        Store the coefficients in self.coeff
        """
        A = np.ones((len(x), self.k+1))
        for row in range(len(x)):
            A[row, 1:] = x[row]

        for sq in range(self.k+1):
            for row in range(len(x)):
                A[row, sq] = A[row, sq] ** sq

        A_inv=np.linalg.pinv(A)
        self.coeff = np.dot(A_inv , y)
        return self.coeff

    def predict(self, x):
        """
        Predict the output given x using the learned coeffecients in
        self.coeff
        """
        temp=self.coeff
        A = np.ones((len(x), self.k + 1))
        for row in range(len(x)):
            A[row, 1:] = x[row]

        for sq in range(self.k + 1):
            for row in range(len(x)):
                A[row, sq] = A[row, sq] ** sq
        outputY=np.dot(A,temp)
        return outputY
        # raise NotImplementedError
LeastSquares(3)