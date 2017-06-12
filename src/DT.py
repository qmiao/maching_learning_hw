import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import mode


class DecisionTree():
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = Node(depth=1)

    def fit(self, X, y):
        data = X
        data['y'] = y
        self.root.split(data, self.max_depth)

    def predict(self, T):
        T['predicted_y'] = -1
        for index, row in T.iterrows():

            current_node = self.root
            while not current_node.is_leaf:
                if row[current_node.column] <= current_node.threshold:
                    next_node = current_node.left_node
                else:
                    next_node = current_node.right_node
                current_node = next_node
            T.set_value(index, 'predicted_y', current_node.classification)
        return T['predicted_y']

    def print_as(self):
        current_level = [self.root]
        while current_level:
            next_level = list()
            for node in current_level:
                print(str(node))
                if node.left_node:
                    next_level.append(node.left_node)
                if node.right_node:
                    next_level.append(node.right_node)
            current_level = next_level


class Node:
    def __init__(self, depth, column=None, threshold=None, is_binary=None, left_node=None, right_node=None,
                 is_leaf=False):
        self.column = column
        self.threshold = threshold
        self.is_binary = is_binary
        self.is_leaf = is_leaf

        if not self.is_leaf:
            self.left_node = left_node
            self.right_node = right_node
        self.depth = depth

    def classify_leaf_node(self, data):
        if self.is_leaf:
            return mode(data['y']).mode[0]
        else:
            return -1

    def split(self, data, max_depth):
        """
        splits a Node based on the data passed in
        """

        # if the Node being split has the same depth as max_depth, no more
        # splitting, declare the Node a leaf node, make a classification decision
        if self.depth == max_depth:
            self.is_leaf = True
            self.classification = self.classify_leaf_node(data)
            return

        data_rows, data_cols = data.shape
        # if the data being passed is empty,

        # if the data being passed has the same classification,
        if len(data['y'].unique()) == 1:
            self.is_leaf = True
            self.classification = self.classify_leaf_node(data)
            return

        # if the data being passed has no attribute left
        # i.e. the data has only 1 column left
        if data_cols == 1:
            self.is_leaf = True
            self.classification = self.classify_leaf_node(data)
            return

        # else, recursively build the tree
        self.column, self.threshold, self.is_binary, left_data, right_data = optimal_test(data)

        # if after calling optimal_test, column is None,
        # i.e. the data being passed does not have a good split point
        # i.e. splitting by any of the attributes in data doesn't make sense
        # then, just declare the node as a leaf node and make classification
        if self.column is None:
            self.is_leaf = True
            self.classification = self.classify_leaf_node(data)
            return

        self.left_node = Node(depth=self.depth + 1)
        self.right_node = Node(depth=self.depth + 1)

        self.left_node.split(left_data, max_depth)
        self.right_node.split(right_data, max_depth)

    def __str__(self):
        self.description = "Node Depth: " + str(self.depth) + "\n"
        if self.is_leaf:
            self.description += "Leaf Node\n"
            self.description += "Classification: " + str(self.classification) + "\n"
        else:
            self.description += "Internal Node\n"
            if self.is_binary:
                self.description += "Test: column " + str(self.column) + " == " + str(self.threshold) + "?\n"
            else:
                self.description += "Test: column " + str(self.column) + " <= " + str(self.threshold) + "?\n"
        return self.description


def validation_curve():
    """
    Reads in a file from the current directory called arrhythmia.csv
    Creates a file called validation.pdf in the current directory with the required plot
    :return:
    """

    data = pd.read_csv("arrhythmia.csv", na_values='?', header=None)

    # shuffle data and divide into 3 equal size data sets
    data1 = data.sample(frac=(1 / 3))
    temp = data.loc[~data.index.isin(data1.index)]
    data2 = temp.sample(frac=(1 / 2))
    data3 = temp.loc[~temp.index.isin(data2.index)]

    max_depth = [i for i in range(2, 21) if i % 2 == 0]  # max_depth = [2, 4, 6, ... , 20]
    avg_training_accuracy = []
    avg_test_accuracy = []

    for depth in max_depth:

        dt1 = DecisionTree(max_depth=depth)
        dt2 = DecisionTree(max_depth=depth)
        dt3 = DecisionTree(max_depth=depth)

        train_data1 = pd.concat([data1, data2])
        train_data2 = pd.concat([data1, data3])
        train_data3 = pd.concat([data2, data3])

        test_data1 = data3
        test_data2 = data2
        test_data3 = data1

        # build the 3 decision trees based on the 3 different training data sets
        dt1.fit(train_data1.iloc[:, :-1], train_data1.iloc[:, -1])
        dt2.fit(train_data2.iloc[:, :-1], train_data2.iloc[:, -1])
        dt3.fit(train_data3.iloc[:, :-1], train_data3.iloc[:, -1])

        # predict the output of the 3 training sets using the 3 decision trees built
        predict_train1 = dt1.predict(train_data1.iloc[:, :-1])  # will return the output Data Series with indices
        predict_train2 = dt2.predict(train_data2.iloc[:, :-1])
        predict_train3 = dt3.predict(train_data3.iloc[:, :-1])
        actual_train1 = train_data1.iloc[:, -1]
        actual_train2 = train_data2.iloc[:, -1]
        actual_train3 = train_data3.iloc[:, -1]

        correct_train1 = 0
        for index in actual_train1.index:
            if predict_train1[index] == actual_train1[index]:
                correct_train1 += 1
        train_accuracy1 = correct_train1 / len(actual_train1)

        correct_train2 = 0
        for index in actual_train2.index:
            if predict_train2[index] == actual_train2[index]:
                correct_train2 += 1
        train_accuracy2 = correct_train2 / len(actual_train2)

        correct_train3 = 0
        for index in actual_train3.index:
            if predict_train3[index] == actual_train3[index]:
                correct_train3 += 1
        train_accuracy3 = correct_train3 / len(actual_train3)

        predict_test1 = dt1.predict(test_data1.iloc[:, :-1])
        predict_test2 = dt2.predict(test_data2.iloc[:, :-1])
        predict_test3 = dt3.predict(test_data3.iloc[:, :-1])
        actual_test1 = test_data1.iloc[:, -1]
        actual_test2 = test_data2.iloc[:, -1]
        actual_test3 = test_data3.iloc[:, -1]

        correct_test1 = 0
        for index in actual_test1.index:
            if predict_test1[index] == actual_test1[index]:
                correct_test1 += 1
        test_accuracy1 = correct_test1 / len(actual_test1)

        correct_test2 = 0
        for index in actual_test2.index:
            if predict_test2[index] == actual_test2[index]:
                correct_test2 += 1
        test_accuracy2 = correct_test2 / len(actual_test2)

        correct_test3 = 0
        for index in actual_test3.index:
            if predict_test3[index] == actual_test3[index]:
                correct_test3 += 1
        test_accuracy3 = correct_test3 / len(actual_test3)

        avg_training_accuracy.append(np.average([train_accuracy1, train_accuracy2, train_accuracy3]))
        avg_test_accuracy.append(np.average([test_accuracy1, test_accuracy2, test_accuracy3]))

    plt.plot(max_depth, avg_test_accuracy, label='test accuracy', lw=2, marker='o')
    plt.plot(max_depth, avg_training_accuracy, label='train accuracy', lw=2, marker='s')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig("validation.pdf")


def optimal_test(data):
    """
    Returns the node in dataframe X_y which gives the largest info gain
    data must have at least 2 columns
    """

    max_info_gain = float("-inf")

    # the values to be returned for this function
    column = None  # the column in X_y with the largest info gain
    threshold = None  # the value at which column should be split
    is_binary = None  # whether column is binary
    left_data = None
    right_data = None  # the two data frames after the split

    # iterate over all attributes and find the one with maximum info gain
    for col in data.iloc[:, :-1]:
        # fill in the missing values of a column
        data[col].fillna(mode(data[col]).mode[0], inplace=True)

        # check whether the attribute has same value, or binary, or numeric
        if len(data[col].unique()) == 1:
            # if the attribute just has 1 value, no split is needed
            continue
        elif len(data[col].unique()) == 2:
            binary = True

            v1 = data[col].unique()[0]
            v2 = data[col].unique()[1]  # the two binary values of this attribute
            left = (data.loc[(data[col] == v1)]).drop([col], axis=1)
            right = (data.loc[(data[col] == v2)]).drop([col], axis=1)
            split_value = v1
            info_gain = entropy(data) - np.average((entropy(left), entropy(right)), weights=(len(left), len(right)))
        else:
            binary = False
            split_value, info_gain, left, right = best_split_value(data, col)

            # if best_split_value returned a negative number for info gain,
            # means that best_split_value could not find a good split value for this attribute

        if info_gain > max_info_gain:
            max_info_gain = info_gain
            column = col
            threshold = split_value
            is_binary = binary
            left_data = left
            right_data = right

    # column, threshold, is_binary, left_data, right_data can be all None
    # if all attributes don't have a good split point
    return column, threshold, is_binary, left_data, right_data


def entropy(data):
    """
    data must have at least 2 columns.
    the last column is the y column
    """
    probs = data.iloc[:, -1].value_counts() / len(data)
    return sum(probs * np.log2(1 / probs))


def best_split_value(data, column):
    """
    finds the best split value for a numeric attribute.
    returns the split value, i.e. threshold, the information gain, and the two
    data frames after the split

    this method will only be called when the attribute has more than 2 unique values
    it's likely that this method will return None, i.e. info_gain is -inf, which means that there are no
    potential split points in this attribute.

    i.e. if max_info_gain is negative infinity, split_value, left_data, right_data are None
    """

    data.sort_values([column], inplace=True)

    row_iterator = data.iterrows()
    _, last = next(row_iterator)

    i = 0

    max_info_gain = float("-inf")
    split_value = None
    left_data = None
    right_data = None

    for _, row in row_iterator:
        if row[column] != last[column]:
            if row[-1] != last[-1]:
                left = (data.iloc[:(i + 1), :]).drop([column], axis=1)
                right = (data.iloc[(i + 1):, :]).drop([column], axis=1)

                # it's possible that left or right would be empty, in such case, the info gain will return 0
                info_gain = entropy(data) - np.average((entropy(left), entropy(right)), weights=(len(left), len(right)))

                if info_gain > max_info_gain:
                    split_value = (row[column] + last[column]) * (1 / 2)
                    max_info_gain = info_gain
                    left_data = left
                    right_data = right

        last = row
        i += 1
    return split_value, max_info_gain, left_data, right_data


validation_curve()
