import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import train_test_split

"""
Implement Artificial Neural Network
"""
class ANN:
    def __init__(self, h, s):
        """
        h: number of hidden layers
        s: number of hidden units in each hidden layer 
        """
        self.h = h
        self.s = s
        # W will be a list of size (h+1), each element is the weight matrix between two layers
        # layers will be indexed from 0 (the input layer) to h+1 (the output layer)
        self.W = list()
        
        
    def fit(self, X, y, alpha, t):
        """
        Trains the network using back propagation, use the logistic activation
        function and the square loss, along with bias inputs
        X: an (m, n)-shaped numpy input matrix
        y: an (m, 1)-shaped numpy output vector
        alpha: training parameter
        t: number of iterations
        """

        m = X.shape[0]        # number of examples
        n = X.shape[1]        # number of features
        
        normalized_X = normalize(X) # normalize X for accuracy
        
        K = len(np.unique(y)) # number of units in the output layer = number of classes
        if K == 2:
            K = 1  # if there are 2 unique values in y, then it's a binary case, just set K to 1
        
        # the Y matrix: each entry (i,j) is either 0 or 1, indicating whether
        # the i'th example belongs to class j
        Y = np.zeros((m, K))
        for i in range(m):
            Y[i][y[i]] = 1
        
        # initiates the weight matrices in W as random small numbers based on Xavier initialization
        # the first weight matrix is between layer 0 (input layer) and layer 1 (first hidden layer)
        # the initial values should be random between [-epsilon_h, epsilon_h]
        epsilon_h = math.sqrt(6/(n + self.s))
        self.W.append(np.random.rand(self.s, (n+1)) * (2 * epsilon_h) - epsilon_h)
        
        # the weight matrices between those hidden layers should have random values between
        # [-epsilon, epsilon]
        epsilon = math.sqrt(6/(self.s + self.s))
        for i in range(self.h-1):
            self.W.append(np.random.rand(self.s, (self.s+1)) * (2 * epsilon) - epsilon)
        
        # the last weight matrix (between the last hidden layer and the output layer)
        # should have random values between [-epsilon_0, epsilon_0]
        epsilon_0 = math.sqrt(6/(self.s + K))
        self.W.append(np.random.rand(K, (self.s+1)) * (2 * epsilon_0) - epsilon_0)


        # a is a list of size (h+2), each element is the array of the activation values for each layer
        a = [None] * (self.h+2)
        # z is a list of size (h+2), each element is the array of the input values for each layer
        # z[0] should be None and never be accessed
        z = [None] * (self.h+2)
        # d is a list of size (h+2), each element is the array of the delta values for each layer
        # d[0] should be None and never be accessed
        d = [None] * (self.h+2)
        
        # the two lists are for plotting cost vs. iterations
        num_iterations = list()
        cost_functions = list()
        
        num_iteration = 0
        # repeat:
        for iteration in range(t):
            # for each example:
            for row_index in range(m):

                # propagate the inputs forward to compute the output
                a[0] = np.insert(normalized_X[row_index], 0, 1)  # shape = (n+1, )
                for l in range(1, self.h+2):  # l goes from 1 to h+1
                    z[l] = np.matmul(self.W[l-1], a[l-1])  # when l = 1, shape = (s, )
                    
                    # add bias term to 'a' only for the hidden layers
                    if l != (self.h+1):
                        a[l] = np.insert(g(z[l]), 0, 1)   # shape = (s+1, )
                    else:
                        a[l] = g(z[l]) # shape = (K, )
                
                # calculates the loss for this example (i.e. in this iteration)
                loss = np.mean((a[self.h+1] - Y[row_index]) ** 2) / 2.0
                
                
                # propagate deltas backward from output layer to input layer
                d[self.h+1] = np.multiply(g_prime(z[self.h+1]), (Y[row_index] - a[self.h+1])) # shape = (K, )
                
                for l in range(self.h, 0, -1):
                    # when l = h, a[l]: shape = (s+1, )
                    #             1-a[l]: shape = (s+1, )
                    #             W[l].T: shape = (s+1, K)
                    #             d[l+1]: shape = (K, )
                    d[l] = np.multiply(np.multiply(a[l], (1-a[l])), np.matmul(self.W[l].T, d[l+1]))[1:] # when l = h, shape = (s, )
                
                # update every weight in network using deltas
                for l in range(self.h+1): # l goes from 0 to h
                    self.W[l] = np.add(self.W[l], alpha * np.outer(d[l+1], a[l]))
                    
                # update the two lists for plotting cost vs. iterations
                num_iterations.append(num_iteration)
                cost_functions.append(loss)
                num_iteration += 1
                
        # plot costs vs. iterations after fitting
        plt.plot(num_iterations, cost_functions)
        plt.ylim([0, 0.25])
        plt.xlabel("iterations")
        plt.ylabel("cost function")
        plt.title("Cost vs. Iterations (alpha=" + str(alpha) + ", h=" + str(self.h) + 
                                        ", s=" + str(self.s) + ", t=" + str(t) + ", " + 
                                        str(m) + " examples, " + str(n) + " features)")
        plt.show()
            
    def predict(self, T):
        """
        Returns the class probabilities for a (q, n)-shaped numpy array T of test examples.
        Assume T has the same columns as X used in training.
        Return value should be an (q, k)-shaped numpy array, P, in which k is the number
        of distinct classes in y during training
        P[i, j] is the model's probability of example i belonging to class j
        """
        # a is a list of size (h+2), each element is the array of the activation values for each layer
        a = [None] * (self.h+2)
        # z is a list of size (h+2), each element is the array of the input values for each layer
        # z[0] should be None and never be accessed
        z = [None] * (self.h+2)
        
        K = self.W[self.h].shape[0] # number of classes
        q = T.shape[0]              # number of examples in test
        P = np.zeros((q, K))        # initiates the P matrix to be returned
        
        # forward propogation:
        for row_index in range(q):
            a[0] = np.insert(T[row_index], 0, 1)  # shape = (n+1, )
            for l in range(1, self.h+2):  # l goes from 1 to h+1
                z[l] = np.matmul(self.W[l-1], a[l-1])  # when l = 1, shape = (s, )
                # add bias term to a only for the hidden layers
                if l != (self.h+1):
                    a[l] = np.insert(g(z[l]), 0, 1)   # shape = (s+1, )
                else:
                    a[l] = g(z[l])
            P[row_index] = a[self.h+1]
        return P
            
    def print(self):
        """
        Prints the current weights from the input layer to the output layer
        """
        for l in range(self.h+1):
            print("Weight matrix between layer " + str(l) + " and layer " + str(l+1))
            print(self.W[l])
        
def g(z):
    """
    The sigmoid function
    z: a (m, )-shaped numpy input array
    """
    return 1.0 / (1 + np.exp(-z))

def g_prime(z):
    """
    The derivative of the sigmoid function
    z: a (m, )-shaped numpy input array
    """
    return np.multiply(g(z), 1-g(z))

def normalize(X):
    """
    :param X: an input numpy matrix of shape (m, n)
    :return: a matrix after doing mean-variance normalization to each column of X 
    """
    
    m = X.shape[0]    # number of rows
    n = X.shape[1]    # number of columns
    
    # since I will be normalizing each column of X by iterating through the rows of
    # X transpose, therefore, I initiate normalized_X to be of shape (n, m).
    # After processing, I will return the transpose of normalized_X
    normalized_X = np.empty([n, m])
    
    index = 0
    for column in X.T:
        column = (column - np.mean(column))
        if np.std(column) != 0:
            column = column / np.std(column)

        # after calculating the normalized value for this column, append this 
        # normalized column to normalized_X
        normalized_X[index] = column
        index += 1

    return normalized_X.T


"""
Test on MINST data
"""
def normalize_test(train_X, test_X):
    """
    :param train_X: a numpy matrix of size (m1, n)
    :param test_X: a numpy matrix of size (m2, n)
    :return: normalized test_X by normalizing each column around the mean and
             variance of each corresponding column of train_X
             (for predicing and testing accuracy)
    """
    
    m = test_X.shape[0]    # number of examples
    n = test_X.shape[1]    # number of features
    
    normazlied_text_X = np.empty([n, m])
    
    index = 0
    for column1, column2 in zip(train_X.T, test_X.T):
        column2 = (column2 - np.mean(column1))
        if np.std(column1) != 0:
            column2 = column2 / np.std(column1)
        normazlied_text_X[index] = column2
        index += 1
    return normazlied_text_X.T


df = pd.read_csv('train.csv')
X = np.array(df.iloc[:, 1:])
y = np.array(df['label'])

# select train and test data:
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

# these two normalized data sets are for predicting and calculating the
# train and test accuracy. They are normalized against the same normalizing
# parameters used in training the model
normalized_X_train = normalize_test(X_train, X_train)
normalized_X_test = normalize_test(X_train, X_test)


# 2 lists to keer track of test accuracy and train accuracy with different
# number of units.
accuracy_test = list()
accuracy_train = list()

# Please note that this following part of code for validating will take a sufficient
# amount of time, since it will be fitting a neural network with 1 hidden layer
# and s units with s ranging from 1 to 244, in order to find the number of units
# where the model starts to overfit
# If you would just like to test the accuracy of the ANN implementation, 
# please skip this validating part
# Please see "validation.png" for the generated plot
for s in range(1, 245):
    clf = ANN(1, s)
    clf.fit(X_train, y_train, 0.01, 1)
    P = clf.predict(normalized_X_test)
    y_est = np.empty_like(y_test)
    for i in range(len(y_test)):
        y_est[i] = np.argmax(P[i])
    
    count = 0
    for y_true, y_pred in zip(y_test, y_est):
        if y_true == y_pred:
            count += 1
    accuracy_test.append(count / len(y_test))
    
    
    P = clf.predict(normalized_X_train)
    y_est = np.empty_like(y_train)
    for i in range(len(y_train)):
        y_est[i] = np.argmax(P[i])
    
    count = 0
    for y_true, y_pred in zip(y_train, y_est):
        if y_true == y_pred:
            count += 1
    accuracy_train.append(count / len(y_train))


plt.plot(range(245), accuracy_test, label='test accuracy', lw=2, marker='o')
plt.plot(range(245), accuracy_train, label='train accuracy', lw=2, marker='s')
plt.xlabel('number of units')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

# from the validating above, I did not notice any overfitting with number of units ranging
# from 1 to 244. I ended up picking the number of units which gave me highest test accuracy,
# which was 123. I fitted a model with all the data points using 1 hidden layer
# and 123 units.
# I then submitted the estimation to the test data set provided by Kaggle,
# the accuracy calculated by Kaggle was about 91%.
df_test = pd.read_csv('test.csv')
X_testing = np.array(df.iloc[:, 1:])
clf = ANN(1, 123)
clf.fit(X, y, 0.01, 1)
P = clf.predict(normalize_test(X, X_testing))
y_est = np.empty_like(y_test)
for i in range(len(y_test)):
    y_est[i] = np.argmax(P[i])





