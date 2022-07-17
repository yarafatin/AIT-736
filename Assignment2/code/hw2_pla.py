import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class PLA(object):
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.epochs = epochs
        self.weights = None
        self.lr = learning_rate
        self._bias = 1
        # error collection
        self.errors = []

    # fit training data
    def fit(self, X, y):
        # Initialize weights as numpy array with zeroes
        weights = np.zeros(X.shape[1])
        self.weights = np.insert(weights, 0, self._bias, axis=0)
        # training here
        track_epoch = 0
        for i in range(self.epochs):
            errors = 0
            for xi, y_target in zip(X, y):
                ws = self.weighted_sum(xi)  # weighted sum
                actual_y = self.activation_function(ws)  # activation function
                # multiply with learning rate - controlled incremental adjustments to weights.
                err = self.lr * (y_target - actual_y)

                # Update weights - back propagation
                self.weights[1:] += err * xi
                self.weights[0] += err

                errors += int(err != 0.0)

            self.errors.append(errors)
            if not errors:
                break
            track_epoch = i
        print('total epochs taken ', track_epoch)

    def weighted_sum(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation_function(self, X):
        # classes are 1 and 0, hence the values. can be made generic to handle any class value
        return np.where(X >= 0, 1, 0)

    def predict(self, X):
        actual_y = np.zeros(X.shape[0], )
        for i, xi in enumerate(X):
            actual_y[i] = self.activation_function(self.weighted_sum(xi))
        return actual_y

    def score(sef, predictions, labels):
        return accuracy_score(labels, predictions)


def plot_decision_boundary(inputs, targets, weights):
    # fig config
    plt.figure(figsize=(10, 6))
    plt.grid(True)

    # plot input samples(2D data points) and i have two classes.
    # one is +1 and second one is -1, so it red color for +1 and blue color for -1
    for input, target in zip(inputs, targets):
        plt.plot(input[0], input[1], 'ro' if (target == 1.0) else 'bo')

    # Here i am calculating slope and intercept with given three weights
    for i in np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1])):
        slope = -(weights[0] / weights[2]) / (weights[0] / weights[1])
        intercept = -weights[0] / weights[2]

        # y =mx+c, m is slope and c is intercept
        y = (slope * i) + intercept
        plt.plot(i, y, 'ko')
    plt.show(block=True)


def main():
    # step 1: Generate data blobs. The classes are 1 and 0
    data_X, data_y = make_blobs(n_samples=1000, cluster_std=1.5, centers=2, n_features=2, random_state=6)
    # stepl 2: split into training and testing 80/20
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.20, stratify=data_y)
    # step 3:Create an instance of our Perceptron
    perceptron = PLA()
    # step 4: Fit the data, display and display our accuracy score
    perceptron.fit(train_X, train_y)
    # step 5: predict test data
    preds = perceptron.predict(test_X)
    # step 6: print accuracy
    accuracy = perceptron.score(preds, test_y)
    print('score obtained', accuracy)
    # step 7: plot decision boundary
    plot_decision_boundary(test_X, preds, perceptron.weights)


if __name__ == '__main__':
    main()
