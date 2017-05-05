import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

mpl.rc('figure', figsize=[10,6])
df = pd.read_csv('wdbc.data', header=None)
base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav', 
                 'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names
df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')


"""
Implement Gradient Descent for Logistic Regression
"""
def gradient_descent(X, y, alpha, T):
    """
    :param X: an input numpy matrix of shape (m, n)
    :param y: a output vector of shape (m, 1)
    :param alpha: a scalar learning rate alpha
    :param T: number of iterations
    :return: a vector theta of shape (n+1, 1), the logistic regression parameter vector
             also plot the value of the cost (loss) function J(theta) vs. the iteration number
    """

    m = X.shape[0]                     # number of examples
    n = X.shape[1]                     # number of features

    normalized_X = normalize(X)        # do mean-variance normalization for the input matrix X
    
    # add a column of 1's to the beginning of normalized_X
    normalized_X = np.insert(normalized_X, 0, 1, axis=1)
    
    theta = np.zeros(n+1)              # initalize theta to 0

    # keep track of the number of iterations and the cost after each iteration for plotting
    num_iterations = range(1, (T+1))
    cost_functions = list()

    # starts iterating to update theta
    for num_iteration in num_iterations:
        
        # update rule:
        # theta := theta + (alpha/sample_size) * (y - h(x)) * x           
        h_x = 1 / (1 + np.exp(-np.matmul(normalized_X, theta)))
        diff_y = y - h_x
        gradient = np.matmul(normalized_X.T, diff_y)
        theta = theta + (alpha/m) * gradient
        
        # after updating theta, calculate the new cost J_theta
        # J_theta = -(1/sample_size) * (sum{y * log(h(x)) + (1-y) * log(1 - h(x))})
        log_h = np.log(1 / (1 + np.exp(-np.matmul(normalized_X, theta))))
        log_1_minus_h = np.log(1 - (1 / (1 + np.exp(-np.matmul(normalized_X, theta)))))
        J_theta = -(1/m) * (np.dot(y, log_h) + np.dot((1-y), log_1_minus_h))

        cost_functions.append(J_theta)

    # after T iterations, plot costs vs. number of iteratons, also returns theta        
    plt.plot(num_iterations, cost_functions)
    plt.xlabel("iterations")
    plt.ylabel("cost function")
    plt.title("Cost vs. Iterations (alpha=" + str(alpha) + ", T=" + str(T) + ", " + 
        str(m) + " samples, " + str(n) + " features)")
    plt.show()
    return theta

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
Fit a Logistic Regression Model on the given data set using Gradient Descent above
"""
X = np.array(df.iloc[:, 2:-1])
y = np.array(df['color'])
gd_theta = gradient_descent(X, y, 4.5, 2000)
print("theta's returned by gradient_descent with alpha=4.5, T=2000 for 30 features")
for theta in gd_theta:
    print(theta)

"""
Problem 7:
Fit a Logistic Regression Model on the give data set using sklearn.linear_model.LogisticRegression
"""
def sk_theta(X, y):
    """
    :param X: an input numpy matrix of shape (m, n)
    :param y: a output vector of shape (m, 1)
    :return: the theta values returned by fitting sklearn.linear_model.LogisticRegression 
             on X (normalized) and y without regularization
    """
    logreg = LogisticRegression(C=1e10)
    
    # Since I would like to compare the returned theta with the theta returned by
    # my gradient_descent function, which takes takes normalized X as input,
    # So I am fitting sklearn.linear_model.LogisticRegression on normalized X here.
    logreg.fit(normalize(X), y)
    theta = np.concatenate([logreg.intercept_, logreg.coef_.ravel()])
    return theta

# Now compare the theta values returned:
# print theta returned by gradient_descent and sklearn side by side
print("theta's (gradient_descent) vs. theta's (sklearn) for 30 features")
for theta1, theta2 in zip(gd_theta, sk_theta(X, y)):
    print(theta1, theta2)
    
# The printed result above looks quite different. However, if I only try a smaller
# number of features such as 10, the result would be quite similar.
# For example, if I take the first 10 features, then compare the theta values for 10 features:
X_10features = np.array(df.iloc[:, 2:12])
gd_theta = gradient_descent(X_10features, y, 10, 20000)
# print theta returned by gradient_descent and sklearn side by side
print("theta's (gradient_descent) vs. theta's (sklearn) for the first 10 features")
for theta1, theta2 in zip(gd_theta, sk_theta(X_10features, y)):
    print(theta1, theta2)
# The printed result above is much closer


# I tried validating my gradient_descent by taking (2/3) of the data as training
# set, and predict the rest (1/3) of the data
def validate(train_X, train_y, test_X, test_y, alpha=4, T=1600):
    
    # get the theta from training set by running gradient_descent
    gd_theta = gradient_descent(train_X, train_y, alpha, T)
    
    # predict class on the test set
    # Since my gradient_descent is returning theta for the normalized data set,
    # so when predicting a test set, I need to normalize it in the same way I
    # normalized the training set, which is what normalize_test() is doing.
    # Then I add a column of 1's to the beginning of test_X,
    # which all together became test_X_modified
    test_X_modified = np.insert(normalize_test(train_X, test_X), 0, 1, axis=1)
    
    # the predicted y value by gradient_descent for the test set
    gd_prob = 1 / (1 + np.exp(-np.matmul(test_X_modified, gd_theta)))
    gd_predict = np.array([(0 if x < 0.5 else 1) for x in gd_prob])
    

    logreg = LogisticRegression(C=1e10)
    logreg.fit(train_X, train_y)
    # the predicted y value by sklearn.linear_model.LogisticRegression
    sk_predict = logreg.predict(test_X)
    
    
    # Now calculate the accuracy of both predictions
    count = 0
    for y_predict, y in zip(gd_predict, test_y):
        if y_predict == y:
            count += 1
    accuracy = count / test_y.size
    print("Gradient Descent Accuracy:", accuracy)
    
    count = 0
    for y_predict, y in zip(sk_predict, test_y):
        if y_predict == y:
            count += 1
    accuracy = count / test_y.size
    print("sklearn.linear_model.LogisticRegression Accuracy", accuracy)

def normalize_test(train_X, test_X):
    """
    :param train_X: a numpy matrix of size (m1, n)
    :param test_X: a numpy matrix of size (m2, n)
    :return: normalized test_X by normalizing each column around the mean and
             variance of each corresponding column of train_X
    """
    
    m = test_X.shape[0]    # number of examples
    n = test_X.shape[1]    # number of features
    
    normazlied_text_X = np.empty([n, m])
    
    index = 0
    for column1, column2 in zip(train_X.T, test_X.T):
        column2 = (column2 - np.mean(column1))
        if np.std(column2) != 0:
            column2 = column2 / np.std(column1)
        normazlied_text_X[index] = column2
        index += 1
    return normazlied_text_X.T

# sample the training set and test set
train = df.sample(frac=(2/3))
test = df.loc[~df.index.isin(train.index)]

train_X = np.array(train.iloc[:, 2:-1])
train_y = np.array(train['color'])
test_X = np.array(test.iloc[:, 2:-1])
test_y = np.array(test['color'])

# Run the validate() method
print("Validating gradient_descent:")
validate(train_X, train_y, test_X, test_y, 4, 100)
# From the printed result, the accuracy of gradient descent
# by training and testing on the give data sets is about 97.89%.
# The accuracy of sklearn.linear_model.LogisticRegression by
# training and testing on the same data sets is about 96.84%.
# Therefore, although the theta values are quite different when fitting
# on 30 features. the model still makes good predictions.



