'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1e-8):
        """
        Constructor
        """
        self.regLambda = reg_lambda
        self.d = degree
        self.theta = None
        self.train_mean = None
        self.train_std = None
        #TODO

    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        #TODO
        # print(X)
        matrix = np.zeros((X.shape[0], degree))
        for i in range(X.shape[0]):
            for j in range(0, degree):
                matrix[i, j] = X[i,] ** (j + 1)
        # print(matrix)
        return matrix


    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        #TODO
        X_poly = self.polyfeatures(X, self.d)
        self.train_mean = np.mean(X_poly, axis=0)
        self.train_std = np.std(X_poly, axis=0)
        X_poly = (X_poly - self.train_mean) / self.train_std

        n = len(X_poly)
        X_ = np.c_[np.ones([n, 1]), X_poly]
        n, d = X_.shape
        d = d - 1  # remove 1 for the extra column of ones we added to get the original num features

        # construct reg matrix
        reg_matrix = self.regLambda * np.eye(d + 1)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y

        self.theta = np.linalg.pinv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)
        # print(self.theta)


    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        # TODO
        X_poly = self.polyfeatures(X, self.d)
        X_poly = (X_poly - self.train_mean) / self.train_std
        n = len(X_poly)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X_poly]

        # predict
        return X_.dot(self.theta)


#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
    n = len(Xtrain)
    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    for i in range(errorTrain.shape[0]):
        model.fit(Xtrain[:i+2], Ytrain[:i+2])
        Y_train_est = model.predict(Xtrain[:i+2])
        for j in range(Y_train_est.shape[0]):
            errorTrain[i] += (Y_train_est[j] - Ytrain[j])**2/Y_train_est.shape[0]

    for i in range(errorTest.shape[0]):
        model.fit(Xtrain[:i+2], Ytrain[:i+2])
        Y_test_est = model.predict(Xtest)
        for j in range(Y_test_est.shape[0]):
            errorTest[i] += (Y_test_est[j] - Ytest[j])**2/Y_test_est.shape[0]
    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape

    return errorTrain, errorTest
