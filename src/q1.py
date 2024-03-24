'''
    Problem Set 8: Question 1
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file contains the solution to question 1
'''
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from template import ytest
from template import y_model

############################################################################################
########################### get_classification_report_df ###################################
############################################################################################
def get_classification_report_df(ytest, y_model) -> pd.DataFrame:
    ''' 
        Function: get_classification_report_df
        Parameters: 2 arrays
            ytest: the true values of y
            y_model: the predicted values of y
        Returns: 1 pd.DataFrame
            report_df: the model classification report as a pandas dataframe

        This function will create a dataframe of the classification report of the model implemented in tempalte.py
    '''
    # get the classification as a dictionary
    report = classification_report(ytest, y_model, output_dict = True)

    # convert to pandas df
    report_df = pd.DataFrame(report)

    return report_df

############################################################################################
############################# calc_cofficient_of_variation #################################
############################################################################################
def calc_coefficient_of_variation(df: pd.DataFrame, col: pd.Series) -> float:
    ''' 
        Function: calc_cofficient_of_variation
        Parameters: 1 pd.Dataframe, 1 pd.Series
            df: the model classification report df
            col: the column used to compute the cv
        Returns: 1 float
            cv: the coefficient of variation of the support column.

        This function calculate the coefficient of variation of the support column.
    '''
    # tranpose report_df to make a support column
    report_df_transpose = df.T

    # filter out aggregates
    support_figs = report_df_transpose[col].iloc[0:10]

    # compute cv
    cv = (np.std(support_figs)/np.mean(support_figs)) * 100

    return cv

############################################################################################
################################### generate_fig ###########################################
############################################################################################
def generate_fig(df: pd.DataFrame) -> None:
    '''
        Function: generate_fig
        Parameters: 1 pd.Dataframe
            df: a dataframe containing the classification report
        Returns: None

        This function will generate a figure that compare f1-score to the support of the 
        classification class.
    '''
    # transpose the report df to extract columns
    report_df_transpose = df.T

    # plot behavior of f1 with respect to support
    sns.scatterplot(data = report_df_transpose,  
                    x = report_df_transpose['support'].iloc[0:10], 
                    y = report_df_transpose['f1-score'].iloc[0:10], 
                    label = r'$f1 = 2 \cdot \frac{(precision \cdot recalll)}{(precision + recall)}$')
    plt.title('F1 with Respect to Support', weight = 'bold', style = 'italic')
    plt.xlabel('Support', weight = 'bold')
    plt.ylabel('F1-Score', weight = 'bold')
    plt.legend(shadow = True)

    # save fig
    save_path = ('figs/f1_with_respect_support.png')
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

############################################################################################
##################################### MAIN #################################################
############################################################################################

# get the classification report as a string
report = classification_report(ytest, y_model)

# display classification report
print(report)

# get the classifcation report df
report_df = get_classification_report_df(ytest, y_model)

# calulate the coefficient of variation of the support column
cv_support = calc_coefficient_of_variation(report_df, 'support')
print(f'The coefficient of variation of the support column is: {cv_support:.2f}%.')

# calulate the coefficient of variation of the support column
cv_f1 = calc_coefficient_of_variation(report_df, 'f1-score')
print(f'The coefficient of variation of the f1-score column is: {cv_f1:.2f}%.')

# generate scatter plot of f1 with respect to supoort
generate_fig(report_df)







