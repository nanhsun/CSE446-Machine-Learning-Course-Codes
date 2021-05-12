import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist


def gradient_descent(X, y, w, b, step, reg):
    # X is n-by-d
    # y is n-by-1
    # w is d-by-1
    # b is scalar
    w_gradient = 0
    b_gradient = 0

    for i in range(y.shape[0]):
        w_gradient += 1 / y.shape[0] * (-y[i]) * X[i, :].reshape((1, X.shape[1])).T * (
                    1 - 1 / (1 + np.exp((-y[i]) * (b + X[i, :].dot(w)))))
        b_gradient += 1 / y.shape[0] * (-y[i]) * (1 - 1 / (1 + np.exp((-y[i]) * (b + X[i, :].dot(w)))))

    w_gradient += + 2 * reg * w
    w_next = w - step * w_gradient
    b_next = b - step * b_gradient

    return w_next.reshape((w_next.shape[0], 1)), b_next

def stochastic_grad_desc(X, y, w, b, step, reg, batch=1):
    w_gradient = 0
    b_gradient = 0
    selected = []

    for i in range(batch):
        selected.append(np.random.randint(0, X.shape[0]))
    for i in selected:
        w_gradient += 1 / len(selected) * (-y[i]) * X[i, :].reshape((1, X.shape[1])).T * (
                    1 - 1 / (1 + np.exp((-y[i]) * (b + X[i, :].dot(w)))))
        b_gradient += 1 / len(selected) * (-y[i]) * (1 - 1 / (1 + np.exp((-y[i]) * (b + X[i, :].dot(w)))))

    w_gradient += 2 * reg * w
    w_next = w - step * w_gradient
    b_next = b - step * b_gradient

    return w_next.reshape((w_next.shape[0], 1)), b_next


### read MNIST
(X_train, labels_train), (X_test, labels_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_train = X_train / 255.0
X_test = X_test / 255.0
not_two_sevens_train = []
not_two_sevens_test = []
for i in range(labels_train.shape[0]):
    if labels_train[i] != 2 and labels_train[i] != 7:
        not_two_sevens_train.append(i)
for i in range(labels_test.shape[0]):
    if labels_test[i] != 2 and labels_test[i] != 7:
        not_two_sevens_test.append(i)
X_train_new = np.delete(X_train, not_two_sevens_train, axis=0)
X_test_new = np.delete(X_test, not_two_sevens_test, axis=0)
labels_train_new = np.delete(labels_train, not_two_sevens_train, axis=0).astype(int)  # y train
labels_test_new = np.delete(labels_test, not_two_sevens_test, axis=0).astype(int)  # y test
for i in range(labels_train_new.shape[0]):
    if labels_train_new[i] == 7:
        labels_train_new[i] = 1
    if labels_train_new[i] == 2:
        labels_train_new[i] = -1
for i in range(labels_test_new.shape[0]):
    if labels_test_new[i] == 7:
        labels_test_new[i] = 1
    if labels_test_new[i] == 2:
        labels_test_new[i] = -1
labels_train_new = labels_train_new.reshape((labels_train_new.shape[0], 1))
labels_test_new = labels_test_new.reshape((labels_test_new.shape[0], 1))
labels_train_new.shape

# b (i)
w = np.zeros((X_train_new.shape[1], 1))
b = 0
step = 0.1
threshold = 0.1
iteration = 0
reg = 0.1
iters = []
J_trains = []
J_tests = []
ws = []
bs = []
J_train = 0
J_test = 0
w_next, b_next = gradient_descent(X_train_new, labels_train_new, w, b, step, reg)
past_w = w_next + 1
past_b = b_next + 1

for i in range(labels_train_new.shape[0]):
    J_train += np.log(1 + np.exp(-labels_train_new[i] * (b_next + X_train_new[i, :].dot(w_next)))) / \
               labels_train_new.shape[0]
for i in range(labels_test_new.shape[0]):
    J_test += np.log(1 + np.exp(-labels_test_new[i] * (b_next + X_test_new[i, :].dot(w_next)))) / \
              labels_train_new.shape[0]

J_train += np.linalg.norm(w_next, ord=2)
J_test += np.linalg.norm(w_next, ord=2)
ws.append(w_next)
bs.append(b_next)
J_trains.append(J_train)
J_tests.append(J_test)
iteration += 1
iters.append(iteration)

while abs(np.amax(w_next - past_w)) >= 0.0003 or abs(b_next - past_b) >= 0.0003:
    print(abs(b_next - past_b))
    J_train = 0
    J_test = 0
    np.copyto(past_w, w_next)
    past_b = b_next
    w_next, b_next = gradient_descent(X_train_new, labels_train_new, w_next, b_next, step, reg)
    iteration += 1
    iters.append(iteration)
    for i in range(labels_train_new.shape[0]):
        J_train += np.log(1 + np.exp(-labels_train_new[i] * (b_next + X_train_new[i, :].dot(w_next)))) / \
                   labels_train_new.shape[0]
    for i in range(labels_test_new.shape[0]):
        J_test += np.log(1 + np.exp(-labels_test_new[i] * (b_next + X_test_new[i, :].dot(w_next)))) / \
                  labels_train_new.shape[0]

    J_train += np.linalg.norm(w_next, ord=2)
    J_test += np.linalg.norm(w_next, ord=2)
    J_trains.append(J_train.item())
    J_tests.append(J_test.item())
    ws.append(w_next)
    bs.append(b_next)
plt.plot(iters, J_trains, label="train")
plt.plot(iters, J_tests, label="test")
plt.xlabel("iteration")
plt.ylabel("J")
plt.legend()
plt.show()

# b (ii)
train_miscounts = []
test_miscounts = []
for j in range(len(bs)):
    trains = []
    tests = []
    test_mis_count = 0
    train_mis_count = 0
    for i in range(X_train_new.shape[0]):
        trains.append(np.sign(bs[j] + X_train_new[i, :].dot(ws[j])))
    for i in range(X_test_new.shape[0]):
        tests.append(np.sign(bs[j] + X_test_new[i, :].dot(ws[j])))
    for i in range(labels_train_new.shape[0]):
        if trains[i] != labels_train_new[i]:
            train_mis_count += 1
    for i in range(labels_test_new.shape[0]):
        if tests[i] != labels_test_new[i]:
            test_mis_count += 1
    train_miscounts.append(train_mis_count / labels_train_new.shape[0])
    test_miscounts.append(test_mis_count / labels_test_new.shape[0])
plt.plot(iters, train_miscounts, label="train error")
plt.plot(iters, test_miscounts, label="test error")
plt.xlabel("iteration")
plt.ylabel("errors")
plt.legend()
plt.show

# c
w = np.zeros((X_train_new.shape[1], 1))
b = 0
step = 0.1
threshold = 0.1
iteration = 0
reg = 0.1
iters = []
J_trains = []
J_tests = []
ws = []
bs = []
J_train = 0
J_test = 0
w_next, b_next = stochastic_grad_desc(X_train_new, labels_train_new, w, b, step, reg)
past_w = w_next + 1
past_b = b_next + 1
iteration += 1
iters.append(iteration)
ws.append(w_next)
bs.append(b_next)
# w_next = sum(ws)/iteration
# b_next = sum(bs)/iteration

for i in range(labels_train_new.shape[0]):
    J_train += np.log(1 + np.exp(-labels_train_new[i] * (b_next + X_train_new[i, :].dot(w_next)))) / \
               labels_train_new.shape[0]
for i in range(labels_test_new.shape[0]):
    J_test += np.log(1 + np.exp(-labels_test_new[i] * (b_next + X_test_new[i, :].dot(w_next)))) / \
              labels_train_new.shape[0]

J_train += np.linalg.norm(w_next, ord=2)
J_test += np.linalg.norm(w_next, ord=2)
# ws.append(w_next)
# bs.append(b_next)
J_trains.append(J_train)
J_tests.append(J_test)
# iteration += 1
# iters.append(iteration)
while abs(np.amax(w_next - past_w)) >= 0.0025 or abs(b_next - past_b) >= 0.0025:

    J_train = 0
    J_test = 0
    np.copyto(past_w, w_next)
    past_b = b_next
    w_next, b_next = stochastic_grad_desc(X_train_new, labels_train_new, w_next, b_next, step, reg)
    iteration += 1
    iters.append(iteration)
    ws.append(w_next)
    bs.append(b_next)
    # w_next = sum(ws)/iteration
    # b_next = sum(bs)/iteration
    for i in range(labels_train_new.shape[0]):
        J_train += np.log(1 + np.exp(-labels_train_new[i] * (b_next + X_train_new[i, :].dot(w_next)))) / \
                   labels_train_new.shape[0]
    for i in range(labels_test_new.shape[0]):
        J_test += np.log(1 + np.exp(-labels_test_new[i] * (b_next + X_test_new[i, :].dot(w_next)))) / \
                  labels_train_new.shape[0]

    J_train += np.linalg.norm(w_next, ord=2)
    J_test += np.linalg.norm(w_next, ord=2)
    J_trains.append(J_train.item())
    J_tests.append(J_test.item())
plt.plot(iters, J_trains, label="train")
plt.plot(iters, J_tests, label="test")
plt.xlabel("iteration")
plt.ylabel("J")
plt.legend()
plt.show()
train_miscounts = []
test_miscounts = []
for j in range(len(bs)):
    trains = []
    tests = []
    test_mis_count = 0
    train_mis_count = 0
    for i in range(X_train_new.shape[0]):
        trains.append(np.sign(bs[j] + X_train_new[i, :].dot(ws[j])))
    for i in range(X_test_new.shape[0]):
        tests.append(np.sign(bs[j] + X_test_new[i, :].dot(ws[j])))
    for i in range(labels_train_new.shape[0]):
        if trains[i] != labels_train_new[i]:
            train_mis_count += 1
    for i in range(labels_test_new.shape[0]):
        if tests[i] != labels_test_new[i]:
            test_mis_count += 1
    train_miscounts.append(train_mis_count / labels_train_new.shape[0])
    test_miscounts.append(test_mis_count / labels_test_new.shape[0])
plt.plot(iters, train_miscounts, label="train error")
plt.plot(iters, test_miscounts, label="test error")
plt.xlabel("iteration")
plt.ylabel("errors")
plt.legend()
plt.show()

# d
w = np.zeros((X_train_new.shape[1], 1))
b = 0
step = 0.5
threshold = 0.1
iteration = 0
reg = 0.1
iters = []
J_trains = []
J_tests = []
ws = []
bs = []
J_train = 0
J_test = 0
w_next, b_next = stochastic_grad_desc(X_train_new, labels_train_new, w, b, step, reg, batch=100)
past_w = w_next + 1
past_b = b_next + 1
iteration += 1
iters.append(iteration)
ws.append(w_next)
bs.append(b_next)
w_next = sum(ws) / iteration
b_next = sum(bs) / iteration

for i in range(labels_train_new.shape[0]):
    J_train += np.log(1 + np.exp(-labels_train_new[i] * (b_next + X_train_new[i, :].dot(w_next)))) / \
               labels_train_new.shape[0]
for i in range(labels_test_new.shape[0]):
    J_test += np.log(1 + np.exp(-labels_test_new[i] * (b_next + X_test_new[i, :].dot(w_next)))) / \
              labels_train_new.shape[0]

J_train += np.linalg.norm(w_next, ord=2)
J_test += np.linalg.norm(w_next, ord=2)
# ws.append(w_next)
# bs.append(b_next)
J_trains.append(J_train)
J_tests.append(J_test)
while abs(np.amax(w_next - past_w)) >= 2e-5 or abs(b_next - past_b) >= 2e-5:

    J_train = 0
    J_test = 0
    np.copyto(past_w, w_next)
    past_b = b_next
    w_next, b_next = stochastic_grad_desc(X_train_new, labels_train_new, w_next, b_next, step, reg, batch=100)
    iteration += 1
    iters.append(iteration)
    ws.append(w_next)
    bs.append(b_next)
    w_next = sum(ws) / iteration
    b_next = sum(bs) / iteration
    for i in range(labels_train_new.shape[0]):
        J_train += np.log(1 + np.exp(-labels_train_new[i] * (b_next + X_train_new[i, :].dot(w_next)))) / \
                   labels_train_new.shape[0]
    for i in range(labels_test_new.shape[0]):
        J_test += np.log(1 + np.exp(-labels_test_new[i] * (b_next + X_test_new[i, :].dot(w_next)))) / \
                  labels_train_new.shape[0]

    J_train += np.linalg.norm(w_next, ord=2)
    J_test += np.linalg.norm(w_next, ord=2)
    J_trains.append(J_train.item())
    J_tests.append(J_test.item())
plt.plot(iters, J_trains, label="train")
plt.plot(iters, J_tests, label="test")
plt.xlabel("iteration")
plt.ylabel("J")
plt.legend()
plt.show()
train_miscounts = []
test_miscounts = []
for j in range(len(bs)):
    trains = []
    tests = []
    test_mis_count = 0
    train_mis_count = 0
    for i in range(X_train_new.shape[0]):
        trains.append(np.sign(bs[j] + X_train_new[i, :].dot(ws[j])))
    for i in range(X_test_new.shape[0]):
        tests.append(np.sign(bs[j] + X_test_new[i, :].dot(ws[j])))
    for i in range(labels_train_new.shape[0]):
        if trains[i] != labels_train_new[i]:
            train_mis_count += 1
    for i in range(labels_test_new.shape[0]):
        if tests[i] != labels_test_new[i]:
            test_mis_count += 1
    train_miscounts.append(train_mis_count / labels_train_new.shape[0])
    test_miscounts.append(test_mis_count / labels_test_new.shape[0])
plt.plot(iters, train_miscounts, label="train error")
plt.plot(iters, test_miscounts, label="test error")
plt.xlabel("iteration")
plt.ylabel("errors")
plt.legend()
plt.show()
