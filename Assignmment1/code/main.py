# HW 1
# 4. Implement basic k-NN classification and the condensed 1-NN algorithm for the Letter Recognition
# Data Set. The first 15,000 examples are for training and the remaining 5,000 for testing. [50% of the
# points]

import datetime
import random as rd
from statistics import mode

import numpy as np
import pandas as pd


# KNN algorithm
def knn_predict(train_x, train_y, test_x, k):
    predicted = []
    # iterate through the test data set to be classified
    for item in test_x:
        distances = []
        # find distance between test data and each individual training Data
        for j in range(len(train_x)):
            distances.append(calculate_distance(np.array(train_x[j]), item))
        distances = np.array(distances)
        # Sort the distances and get the first k record's index
        dist = np.argsort(distances)[:k]
        # get labels of the k data points
        labels = train_y[dist]
        # majority label occurrence
        predicted.append(mode(labels)[0])
    return predicted


# compute distance between two vectors
def calculate_distance(v1, v2):
    # using euclidean distance
    return np.linalg.norm(v1 - v2)


# iterate over mulitple ks to determine the one with best accuracy
def knn_best_k_accuracy(train_x, test_x, train_y, test_y, start_k, max_k):
    for k in range(start_k, max_k, 2):
        actual_y = knn_predict(train_x, train_y, test_x, k)
        accuracy = (test_y == actual_y).sum() / float(test_y.size) * 100
        print('k is {}, score is {}'.format(k, accuracy))


# method to condense data
def condense_data(train_x, train_y, k):
    count = len(train_x)
    # start condensing with the first element
    condensed_x = [train_x[0]]
    condensed_y = [train_y[0]]
    condensed_idx = []
    processed = []
    # iterate through the training data set
    while len(processed) != len(train_x):
        # identify a random index that is in the range of training data set and is not yet processed
        i = rd.choice([x for x in range(count) if x not in processed]) #3
        processed.append(i)
        # perform knn on the condensed dataset as training set and classify remaining training samples
        test_y = knn_predict(np.array(condensed_x), np.array(condensed_y), np.array([train_x[i]]), k)
        # if classification don't match, add to the condensed training set
        if test_y[0] != train_y[i]:
            condensed_x.append(train_x[i])
            condensed_y.append(train_y[i])
            condensed_idx.append(i)
    return np.asarray(condensed_x), np.asarray(condensed_y)


def read_data(split_records):
    df = pd.read_csv('letter-recognition.data', header=None)
    # df = df.iloc[0:10]
    data = df[df.columns[1:]].to_numpy()
    label = df[df.columns[0]].to_numpy()
    train_x, test_x = data[0:split_records, :], data[split_records:]
    train_y, test_y = label[0:split_records], label[split_records:]
    return train_x, test_x, train_y, test_y


def main():
    # tuning params
    split_records = 15000
    start_k = 1
    max_k = 11

    # read data
    train_x, test_x, train_y, test_y = read_data(split_records)

    # ####### KNN ####################
    start_time = datetime.datetime.now()
    print('knn started at:', start_time)
    knn_best_k_accuracy(train_x, test_x, train_y, test_y, start_k, max_k)
    end_time = datetime.datetime.now()
    print('knn ended at {}. It took {} '.format(end_time, end_time - start_time))

    # ####### Condensed 1NN ####################
    k = 1
    start_time = datetime.datetime.now()
    print('Condensed 1-NN started at:', start_time)
    condensed_x, condensed_y = condense_data(train_x, train_y, k)
    print('Condensed training size {}, original size {}'.format(len(condensed_x), len(train_x)))
    actual_y = knn_predict(condensed_x, condensed_y, test_x, k)
    accuracy = (test_y == actual_y).sum() / float(test_y.size) * 100
    print('Condensed 1-NN score is {}'.format(accuracy))
    end_time = datetime.datetime.now()
    print('Condensed 1-NN ended at {}. It took {} '.format(end_time, end_time - start_time))


if __name__ == '__main__':
    main()
