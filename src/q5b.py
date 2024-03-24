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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

##########################################################################################################
##################################### find_best_svm_pca_model ############################################
##########################################################################################################
def execute_best_svm_pca_model() -> None:
    ''' 
        Function: execute_best_svm_pca_model
        Pamameters: None
        Returns: None
    '''
    # load data
    digits = load_digits()
    X, y = digits.data, digits.target

    # seperate data into train/test split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, 
                                                    y,
                                                    random_state = 42)

    # instantiate the models, PCA and SVC
    pca = PCA(n_components = 34,
              whiten = True,
              svd_solver = 'randomized', 
              random_state = 42)
    svc = SVC(kernel = 'rbf', C = 10, gamma = 0.005, random_state = 42)

    # instantiate pipeline
    model = make_pipeline(pca, svc)

    # fit the model
    model.fit(Xtrain, ytrain)

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
    save_path = ('figs/SVM_confusion_5b.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close();

##########################################################################################################
########################################### main #########################################################
##########################################################################################################

execute_best_svm_pca_model()