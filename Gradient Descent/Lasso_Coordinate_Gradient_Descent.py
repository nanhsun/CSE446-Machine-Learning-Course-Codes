import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def lasso_coordinate_descent(X, y, reg, threshold, w_hat=None):
    ## X is n-by-d
    ## y is n-by-1
    if len(X.shape) == 1:
        length = 1
    else:
        length = X.shape[1]
    if w_hat is None:
        w = np.zeros((length, 1))  ## w is d-by-1
        w = w.reshape((length, 1))
    else:
        w = w_hat
    y = y.reshape((X.shape[0], 1))
    past = w + threshold + 1

    while abs(np.amax(w - past)) >= threshold:
        np.copyto(past, w)
        Xw = X.dot(w)  ## Xw is n-by-1
        b = np.sum(y - Xw) / X.shape[0]  ## b is scalar
        a_square = np.sum(np.square(X), axis=0)
        for k in range(length):
            new_X = np.delete(X, k, 1)
            new_w = np.delete(w, k, 0)
            new_Xw = new_X.dot(new_w)
            ak = 2 * a_square[k]
            ck = 2 * X[:, k].reshape((1, X.shape[0])).dot(y - (b + new_Xw))
            if ck < -reg:
                w[k] = (ck + reg) / ak
            if ck >= -reg and ck <= reg:
                w[k] = 0
            if ck > reg:
                w[k] = (ck - reg) / ak
    return w


def lasso_predict(X, w_hat):
    return X.dot(w_hat)


crimetest = pd.read_csv("hw2data/crime-test.csv")
crimetrain = pd.read_csv("hw2data/crime-train.csv")
crimetrain_X = crimetrain.drop(["ViolentCrimesPerPop"], axis=1).to_numpy()
crimetrain_y = crimetrain["ViolentCrimesPerPop"].to_numpy()
crimetest_X = crimetest.drop(["ViolentCrimesPerPop"], axis=1).to_numpy()
crimetest_y = crimetest["ViolentCrimesPerPop"].to_numpy()

crimetrain_y = crimetrain_y.reshape((crimetrain_y.shape[0], 1))
crimetest_y = crimetest_y.reshape((crimetest_y.shape[0], 1))
crimetrain_y_sum = np.sum(crimetrain_y)
reg = np.max(2 * abs((crimetrain_y - crimetrain_y_sum / n).T.dot(crimetrain_X)))
regs = []
non_zeros = []
agePct = []
pctWSoc = []
pctUrban = []
agePct65 = []
household = []
errorTrain = []
errorTest = []
increment = 0.1
w_hat = lasso_coordinate_descent(crimetrain_X, crimetrain_y, reg, increment)
regs.append(reg)
non_zeros.append(np.count_nonzero(w_hat))
agePct.append(w_hat[3][0])
pctWSoc.append(w_hat[12][0])
pctUrban.append(w_hat[7][0])
agePct65.append(w_hat[5][0])
household.append(w_hat[1][0])
Y_train_est = lasso_predict(crimetrain_X, w_hat)
Y_test_est = lasso_predict(crimetest_X, w_hat)

errorTrain.append(np.sum(np.square(Y_train_est - crimetrain_y) / Y_train_est.shape[0]))
errorTest.append(np.sum(np.square(Y_test_est - crimetest_y) / Y_test_est.shape[0]))
while np.count_nonzero(w_hat) < w_hat.shape[0]:
    reg -= 2
    w_hat = lasso_coordinate_descent(crimetrain_X, crimetrain_y, reg, increment, w_hat=w_hat)
    regs.append(reg)
    non_zeros.append(np.count_nonzero(w_hat))
    # print(w_hat.shape)
    agePct.append(w_hat[3][0])
    pctWSoc.append(w_hat[12][0])
    pctUrban.append(w_hat[7][0])
    agePct65.append(w_hat[5][0])
    household.append(w_hat[1][0])

    Y_train_est = lasso_predict(crimetrain_X, w_hat)
    Y_test_est = lasso_predict(crimetest_X, w_hat)

    errorTrain.append(np.sum(np.square(Y_train_est - crimetrain_y) / Y_train_est.shape[0]))
    errorTest.append(np.sum(np.square(Y_test_est - crimetest_y) / Y_test_est.shape[0]))
plt.xscale("log")
plt.plot(regs, non_zeros)
plt.xlabel("regulization parameter")
plt.ylabel("non zero numbers")
# plt.rcParams['figure.figsize'] = [5,5]
plt.show()
plt.xscale("log")
plt.plot(regs, agePct, label="agePct12t29")
plt.plot(regs, pctWSoc, label="pctWSocSec")
plt.plot(regs, pctUrban, label="pctUrban")
plt.plot(regs, agePct65, label="agePct65up")
plt.plot(regs, household, label="householdsize")
plt.xlabel("regulization parameter")
plt.ylabel("w")
plt.legend()
# plt.rcParams['figure.figsize'] = [5,5]
plt.show()
plt.xscale("log")
plt.plot(regs, errorTrain, label="Train Error")
plt.plot(regs, errorTest, label="Test Error")
plt.legend()
plt.xlabel("regulization parameter")
plt.ylabel("error")
plt.show()
## A.5 (f)
crimetrain_y_sum = np.sum(crimetrain_y)
reg = int(np.max(2 * abs((crimetrain_y - crimetrain_y_sum / n).T.dot(crimetrain_X))))
w_hat = lasso_coordinate_descent(crimetrain_X, crimetrain_y, reg, increment)

while reg != 30:
    reg -= 2
    w_hat = lasso_coordinate_descent(crimetrain_X, crimetrain_y, reg, increment, w_hat=w_hat)
maximum_feature = np.where(w_hat == np.max(w_hat))
minimum_feature = np.where(w_hat == np.min(w_hat))
print(maximum_feature)  # PctIlleg
print(minimum_feature)  # PctKids2Par
