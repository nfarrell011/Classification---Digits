'''
    Problem Set 8: Question 5
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file contains the solution to question 5
'''
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

##########################################################################################################
##################################### find_best_svm_pca_model ############################################
##########################################################################################################
def find_best_svm_pca_model() -> None:
    ''' 
        Function: find_best_svm_pca_model
        Pamameters: None
        Returns: None

        This function will perform a GridSearch to find the best PCA SVC model by examining the various
        values of the hyperparamters.

        The hyperparameters used in the GridSearch:
        
            PCA: n_components: (1, 2, ..., 64)

            SVC: kernel: {rbf, poly}
                 C: {1, 10, 50}
                 gamma: {.0001, .0005, .001, 0.005}

        After the best model has been found the best parameters will be displayed.
        
        The best model will be fit to the test data and the classification report displayed.

        Lastly, a confusion matrix of the results will be displayed.

        ************************************ NOTICE ********************************************
        This function executes 1536 models, and will take over 20 minutes execute.
        ****************************************************************************************

        For your convenience, 'q5b.py' will fit and display the results of the best model, without going through
        the GridSearch
    '''
    # load data
    digits = load_digits()
    X, y = digits.data, digits.target

    # seperate data into train/test split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, 
                                                    y,
                                                    random_state = 42)

    # instantiate the models, PCA and SVC
    pca = PCA(whiten = True,
            svd_solver = 'randomized', 
            random_state = 42)
    svc = SVC()

    # set the range of principle components to use in GridSearch
    n_compenents = list(range(1,65))

    # instantiate pipeline
    model = make_pipeline(pca, svc)

    # set the hyperparameters to use in GridSearch
    param_grid = {'pca__n_components': n_compenents, 
                  'svc__kernel': ['rbf', 'poly'],
                  'svc__C': [1, 10, 50],
                  'svc__gamma': [.0001, .0005, .001, 0.005]}

    # instantiate GridSearch
    grid = GridSearchCV(model, param_grid, return_train_score = True)

    # fit the model
    grid.fit(Xtrain, ytrain)

    # display the best parameters 
    print('The best parameters are:', grid.best_params_)

    # exstract the best model
    model = grid.best_estimator_

    # make predictions with test data
    yfit = model.predict(Xtest)

    # print accuracy score
    print(f'The accuracy score of the best model is: {accuracy_score(ytest, yfit):.4f}.')

    # print the classification report
    print(classification_report(ytest, yfit))

    # display confusion matrix
    mat = confusion_matrix(ytest, yfit)
    sns.heatmap(mat.T, 
                square = True, 
                annot = True, 
                fmt = 'd', 
                cbar = False, 
                cmap = 'Reds')
    plt.title('Confusion Matrix ~ PCA SVM', weight = 'bold')
    plt.xlabel('True Label', weight = 'bold')
    plt.ylabel('Predicted Label', weight = 'bold')

    # save fig
    save_path = ('figs/SVM_confusion_5a.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close();

##########################################################################################################
########################################### main #########################################################
##########################################################################################################

find_best_svm_pca_model()