## Ridge Regression Closed Form##

from keras.datasets import mnist
import numpy as np
from scipy import linalg
import math
from matplotlib import pyplot as plt
# from numba import jit, cuda

# @cuda.jit(target="cuda")
def train(X, Y, reg_lambda):
    X_ = X.T.dot(X)
    temp = X_ + reg_lambda * np.eye(X_.shape[0])
    eye = np.eye(X_.shape[0])
    inverted = linalg.solve(temp, eye)

    return inverted.dot(X.T).dot(Y)
# @cuda.jit(target="cuda")
def predict(W, X_prime):
    result = np.zeros(X_prime.shape[0])
    for i in range(X_prime.shape[0]):
        temp = []
        for j in range(10):
            e = np.zeros(W.T.shape[0])
            e[j] = 1.0
            temp.append(e.dot(W.T).dot(X_prime[i, :].T))
        result[i] = temp.index(max(temp))

    return result
# @cuda.jit(target="cuda")
def error(estimate, real_data):
    count = 0
    for i in range(len(estimate)):
        if estimate[i] == real_data[i]:
            count += 1
    return (1-count/len(real_data)) * 100

if __name__=="__main__":
    (X_train, labels_train), (X_test, labels_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train/255.0
    X_test = X_test/255.0

    ### A6 b ###
    labels_train_hot = np.zeros([labels_train.shape[0], 10])
    labels_test_hot = np.zeros([labels_train.shape[0], 10])
    for count, number in enumerate(labels_train):
        labels_train_hot[count, number] = 1
    for count, number in enumerate(labels_test):
        labels_test_hot[count, number] = 1
    w = train(X_train, labels_train_hot, reg_lambda=1e-4)
    X_test_est = predict(w, X_test)
    X_train_est = predict(w, X_train)

    train_error = error(X_train_est, labels_train)
    test_error = error(X_test_est, labels_test)
    print("train error: ", train_error)
    print("test error: ", test_error)

    ### B.2 a ###
    x_and_label = np.c_[X_train.reshape(len(X_train), -1), labels_train.reshape(len(labels_train), -1)]
    X_train2 = x_and_label[:, :X_train.size//len(X_train)].reshape(X_train.shape)
    labels_train2 = x_and_label[:, X_train.size//len(X_train):].reshape(labels_train.shape)
    np.random.shuffle(x_and_label)
    xtraining_partitioned = []
    labels_partitioned = []
    k = 5
    start = 0
    for i in range(int(X_train2.shape[0]/k),X_train2.shape[0]+1, int(X_train2.shape[0]/k)):
        # print(start)
        xtraining_partitioned.append(X_train2[start:i, :])
        labels_partitioned.append(labels_train2[start:i])
        start = i

    d = X_train.shape[1]
    total_train_error_averages = []
    total_val_error_averages = []
    ps = [100]
    for p in ps:
        val_errors = []
        train_errors = []
        b = np.zeros(int(X_train.shape[0]*(k-1)/k))
        for i in range(len(b)):
            b[i] = np.random.uniform(low=0, high=2 * math.pi)
        b = np.array([b, ] * p)


        G = np.zeros((p, d))

        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                G[i, j] = np.random.normal(loc=0.0, scale=0.1)
        #
        for i in range(k):
            print(i)
            xtemp_training_part = []
            temp_labels_part = []
            for j in range(len(xtraining_partitioned)):
                if j == i:
                    continue
                xtemp_training_part.append(xtraining_partitioned[j])
            for j in range(len(labels_partitioned)):
                if j == i:
                    continue
                temp_labels_part.append(labels_partitioned[i])
            xval = xtraining_partitioned[i]
            labelval = labels_partitioned[i]
            xtraining = np.concatenate(xtemp_training_part, axis=0)
            labeltraining = np.concatenate(temp_labels_part, axis=0)
            labeltraining_hot = np.zeros([labeltraining.shape[0], 10])
            for count, number in enumerate(labeltraining):
                labeltraining_hot[count, int(number)] = 1

            h = np.cos(G.dot(xtraining.T) + b)
            w_ = train(h, labeltraining_hot, reg_lambda=0)
            xtraining_est = predict(w_, xtraining)
            validation_est = predict(w_, xval)
            train_errors.append(error(xtraining_est, labeltraining))
            val_errors.append(error(validation_est, labelval))
        total_train_error_averages.append(sum(train_errors)/len(train_errors))
        total_val_error_averages.append(sum(val_errors)/len(val_errors))

    plt.plot(ps, total_train_error_averages, label="train error")
    plt.plot(ps, total_val_error_averages, label="validation error")
    plt.legend()
    plt.xlabel("p values")
    plt.ylabel("error")
