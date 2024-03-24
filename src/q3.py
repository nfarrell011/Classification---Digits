'''
    Problem Set 8: Question 3
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file contains the solution to question 3
'''
from sklearn.datasets import load_digits
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

##########################################################################################################
################################### generate_n_estimators_fig ############################################
##########################################################################################################
def generate_n_estimators_fig():
    ''' 
        Function: generate_n_estimators_fig
        Parameters: None
        Returns: None

        This function will generate a figure that displays how the number of estimators used in 
        random forest classifier impacts model performance.
    '''
    # load the digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # instantaite pipeline
    pcr = make_pipeline(RandomForestClassifier(random_state = 0))

    # set the range of the hyperparameter, n_estimators
    num_estimators = [10, 20, 50, 75, 100, 150, 200, 500, 750, 1000]

    # set the number of folds using shuffle split
    cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 42)

    # instantiate GridSearch
    clf = GridSearchCV(pcr, {'randomforestclassifier__n_estimators': num_estimators}, 
                       cv = cv,
                       refit = True, 
                       return_train_score = True)

    # fit model
    clf.fit(X, y)

    # extract results
    test_scores = clf.cv_results_["mean_test_score"]
    test_scores_std = clf.cv_results_["std_test_score"]
    train_scores = clf.cv_results_["mean_train_score"]
    train_scores_std = clf.cv_results_["std_train_score"]

    # compute standard errors
    test_std_error = test_scores_std / np.sqrt(5)
    train_std_error = train_scores_std / np.sqrt(5)

    # generate figure
    plt.figure().set_size_inches(8, 6)

    # scores
    plt.plot(num_estimators, test_scores, marker = 'o', label = 'Test Scores', color = 'green')
    plt.plot(num_estimators, train_scores, marker = 'o', color = 'red', label = 'Train Scores')

    # standard error
    plt.fill_between(num_estimators, test_scores + test_std_error, test_scores - test_std_error, color = 'green', 
                    alpha = 0.2)
    plt.fill_between(num_estimators, train_scores + train_std_error, train_scores - train_std_error, 
                    alpha = 0.2, color = 'darkorange')

    # max score
    plt.axhline(np.max(test_scores), linestyle = "--", color= 'orange', label = 'Max Score')

    # labels
    plt.title(r'Random Forest Classifier ~ $m = \sqrt{p}$: Accuracy Score'+ '\n' + 'with Respect to Number of Estimators', 
            weight = 'bold', 
            style = 'italic', 
            fontsize = 14)
    plt.ylabel(r'Accuracy Score $\pm SE = \frac{\sigma}{\sqrt{n}}$', fontsize = 14, weight = 'bold')
    plt.xlabel('Number of Estimators ~ Trees', weight = 'bold')
    plt.annotate(f'Max Score = {np.max(test_scores):.3f}', 
                xy = (1.1, .47), 
                xytext=(1.1, .47), 
                weight = 'bold', 
                color = 'red')
    plt.xlim([num_estimators[0], num_estimators[-1]])
    plt.legend(loc = 'best', shadow = True)
    plt.grid()

    # save fig
    save_path = ('figs/n_estimators.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

##########################################################################################################
########################################### main #########################################################
##########################################################################################################

generate_n_estimators_fig()


