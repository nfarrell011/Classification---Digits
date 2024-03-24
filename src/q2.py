'''
    Problem Set 8: Question 2
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file contains the solution to question 2
'''
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier

############################################################################################
############################# generate_learning_curve_plot #################################
############################################################################################
def generate_learning_curve_plot():
    ''' 
        Function: generate_learning_curve_plot
        Parameters: None
        Returns: None

        This function will generate a learning curve figure by 
    '''
    # load the digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # instantiate a DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state = 42)

    # define the cv
    cv = ShuffleSplit(n_splits = 50, test_size = 0.2, random_state = 42)

    # define the range of training set sizes, 10% - 100%
    train_sizes = np.arange(0.1, 1.1, .1)

    # calculate the learning curve scores
    train_sizes, train_scores, test_scores = learning_curve(clf, 
                                                            X, 
                                                            y, 
                                                            train_sizes = train_sizes, 
                                                            cv = cv, 
                                                            scoring = 'accuracy')

    # calculate mean and standard deviation of scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.title('Learning Curve: Decision Tree Classifier', weight = 'bold', style = 'italic', fontsize = 16)
    plt.xlabel('Number of Training Samples', weight = 'bold')
    plt.ylabel('Accuracy', weight = 'bold')

    plt.grid()
    plt.fill_between(train_sizes, 
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, 
                    alpha = 0.1,
                    color = "r")
    plt.fill_between(train_sizes, 
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, 
                    alpha = 0.1, 
                    color = "g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label = "Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label = "Cross-validation score")

    plt.legend(loc="best")

    # save fig
    save_path = ('figs/learning_curve.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

############################################################################################
########################################### main ###########################################
############################################################################################

# generate fig
generate_learning_curve_plot()




