import numpy as np
import scipy.io

data_filename = "ex3data1.mat"
weight_filename = "ex3weights.mat"

# return a dictionary
data = scipy.io.loadmat(data_filename)
weights = scipy.io.loadmat(weight_filename)
# get X and y and weights return ndarray
X = data["X"]
y = data["y"]
theta1 = weights["Theta1"]
theta2 = weights["Theta2"]


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def predict(theta1, theta2, X):
    # add ones to X
    m = X.shape[0]
    ones = np.ones([m, 1])
    # X(5000,401) theta1(25,401)
    X = np.hstack([ones, X])
    # middle_layer(5000,25)
    middle_layer = sigmoid(X.dot(theta1.T))
    # add ones to middle_layer(5000,26)
    ones = np.ones([middle_layer.shape[0], 1])
    middle_layer = np.hstack([ones, middle_layer])
    # middle_layer(5000,26) theta2(10,26)  output(5000,10)
    output = sigmoid(middle_layer.dot(theta2.T))
    # the index of max value in each row is the predict number
    # the index system in python and matlab is different in matlab is start from 1
    # so need to fix the result + 1 to match the y
    result = np.argmax(output, axis=1) + 1
    return result


# predict_result(5000,) y(5000,1)
predict_result = predict(theta1, theta2, X)
# flatten y to (5000,)
y = y.flatten()

correctness = np.sum(predict_result == y) / len(y) * 100
print(correctness)
