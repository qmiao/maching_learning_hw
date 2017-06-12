import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn import tree
from sklearn import linear_model
from sklearn.metrics import make_scorer
import csv


# Reads in the Train and Test data (https://www.kaggle.com/c/prudential-life-insurance-assessment)
train_data = pd.read_csv("prudential_train.csv", index_col=0)
test_data = pd.read_csv("prudential_test.csv", index_col=0)

# Since there are no explicit limits on the number of unique values in categorical variable, discrete variable, etc.
# I hardcoded the list of continuous variables and discrete variables based on the data description.
# Variables not in these two lists are either categorical variables or binary variables (binary variables in this data
# set starts with "Medical_Keyword").
continuous_variable = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4",
                       "Employment_Info_6", "Insurance_History_5", "Family_Hist_2", "Family_Hist_3", "Family_Hist_4",
                       "Family_Hist_5"]
discrete_variable = ["Medical_History_1", "Medical_History_10", "Medical_History_15", "Medical_History_24",
                     "Medical_History_32"]

def impute(dataframe):
    """
    Imputes a datafram with null values
    Strategy:
        - if variable is cateogorical, use mode
        - if variable is continuous, use mean
        - if variable is discrete, use median
    """
    imp_median = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp_mode = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    for col in dataframe.iloc[:, :-1]:
        # if the entire column is null, just delete this column from the dataframe
        if sum(dataframe[col].isnull()) == len(dataframe[col]):
            dataframe.drop([col], axis=1, inplace=True)
        # else, impute the column based on the Strategy above
        elif sum(dataframe[col].isnull()) > 0:
            if col in discrete_variable:
                dataframe[col] = imp_median.fit_transform(dataframe[[col]]).ravel()
            elif col in continuous_variable:
                dataframe[col] = imp_mean.fit_transform(dataframe[[col]]).ravel()
            else:
                dataframe[col] = imp_mode.fit_transform(dataframe[[col]]).ravel()

def create_dummy(dataframe):
    """
    Creates dummy variables for categorical variables
    """
    X = dataframe.drop('Response', axis=1)
    y = dataframe['Response']

    # constructs a new dataframe X_new to store the dummy variables and the non-categorical variables
    # in the original data set
    X_new = pd.DataFrame(index=X.index)
    for col in X:
        if not col.startswith("Medical_Keyword"):
            if col not in (discrete_variable + continuous_variable):
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X_new = pd.concat([X_new, dummies], axis=1)
            else:
                X_new = pd.concat([X_new, X[col]], axis=1)
        else:
            X_new = pd.concat([X_new, X[col]], axis=1)
    return pd.concat([X_new, y], axis=1)



def plot_validation_curve(estimator, title, X, y, param_name, param_range, logx):
    """
    Plot validation curve and learning curve for decision tree model and logistic model
    """

    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name,
                                                 param_range=param_range, cv=5,
                                                 scoring=make_scorer(cohen_kappa_score, weights="quadratic"))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    lw = 2
    if logx:
        plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    else:
        plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    if logx:
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    else:
        plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
    plt.legend(loc="best")
    return plt


def plot_learning_curve(estimator, title, X, y):
    """
    Plot learning curve for decision tree model and logistic model
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5,
                                                            train_sizes=np.linspace(.1, 1.0, 5),
                                                            scoring=make_scorer(cohen_kappa_score, weights="quadratic"))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt


"""
Decision Tree Model:
"""
# Put the train and test data in one data frame, impute and create dummy variables
total_data = train_data.append(test_data)
impute(total_data)
new_total = create_dummy(total_data)

# Re-separate train and test data
train = new_total.loc[(new_total.index.isin(train_data.index))]
test = new_total.loc[(new_total.index.isin(test_data.index))]

# cross validate decision tree model for training data to find the best max_depth
sample = train.sample(frac=(1/5))
X = sample.drop('Response', 1)
y = sample['Response']

# Plot the validation curve for DecisionTreeClassifier with parameter max_depth
param_range = [i for i in range(2, 21) if i % 2 == 0]
title = "Validation Curve with DecisionTreeClassifier"
plot_validation_curve(tree.DecisionTreeClassifier(), title, X, y, "max_depth", param_range, False)
plt.show()

# From the validation curve plotted, when max_depth = 10, Cross-validation score is the highest.
# Plot the learning curve for DecisionTreeClassifier with max_depth=10
title = "Learning Curves (DecisionTreeClassifier with max_depth=10)"
plot_learning_curve(tree.DecisionTreeClassifier(max_depth=10), title, X, y)
plt.show()

# seems like max_depth = 10 is the best
# starts to fit a model on train and predict on test
dt_model = tree.DecisionTreeClassifier(max_depth=10)
dt_model.fit(train.drop('Response', 1), train['Response'])
predictions = dt_model.predict(test.drop('Response',1))
indices = np.array(test.index)

# write the predictions to a csv file: Kaggle result: 0.48
with open('sample_submission.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Response']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ind, pred in zip(indices, predictions):
        writer.writerow({'Id': ind, 'Response': pred})



"""
Logistic Regression Model:
"""
# cross validate logistic regression model for training data to find the best C value
# Plot the validation curve the LogisticRegression with parameter C
param_range = [10 ** i for i in range(-6, 4)]
title = "Validation Curve with Logistic Regression"
plot_validation_curve(linear_model.LogisticRegression(), title, X, y, "C", param_range, True)
plt.show()

# From the validation curve plotted, when C = 1, Cross-validation score is the highest.
# Plot the learning curve for LogisticRegression with C=1
title = "Learning Curves (LogisticRegression with C=1)"
plot_learning_curve(linear_model.LogisticRegression(C=5), title, X, y)
plt.show()

# Fit a logistic regression model
lr_model = linear_model.LogisticRegression()
lr_model.fit(train.drop('Response', 1), train['Response'])
predictions = lr_model.predict(test.drop('Response',1))
indices = np.array(test.index)

# write the predictions to a csv file: Kaggle Score: 0.51925
with open('prudential_submission.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Response']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    
    for ind, pred in zip(indices, predictions):
        writer.writerow({'Id': ind, 'Response': pred})
