'''
    Problem Set 8: Question 4
    Joseph Farrell
    DS 5110 Intro to Data Management
    10/29/2023

    This file contains the solution to question 4
'''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

##########################################################################################################
####################################### generate_adaboost_fig ############################################
##########################################################################################################
def generate_adaboost_fig(save_path: str, learning_rate: float = .001, n_estimators: int = 100) -> None:
    ''' 
        Function: generate_adaboost_fig
        Parameters: 1 string, 1 float, 1 int
            save_path: path to the saving location
            learning_rate: learning rate of the AdaBoostClassifier
            n_estimators: number of trees in AdaBoostClassifier
        Returns: None

        This function will generate a figure exploring the AdaBoostClassifier over different values of the 
        hyperparamters. Learning rate and n_estimators can be set as arguments. Depth will range from 1 to 10.
    '''
    # load data
    digits = load_digits()
    X, y = digits.data, digits.target

    # split the data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.2, 
                                                        shuffle = True)

    # set the range of the depth
    max_depths = list(range(1, 11, 2))

    # instantiate list to store accuracy scores by depth
    accuracy_by_depth = []

    # iterate over the different depths
    for depth in max_depths:

        # set the stump
        dt_stump = DecisionTreeClassifier(max_depth = depth, min_samples_leaf = 1, random_state = 0)
        dt_stump.fit(X_train, y_train)

        # instantiate AdaBoostClassifier
        ada_discrete = AdaBoostClassifier(estimator = dt_stump,
                                         learning_rate = learning_rate,
                                         n_estimators = n_estimators,
                                         algorithm = 'SAMME',
                                         random_state = 0)
        
        # fit model
        ada_discrete.fit(X_train, y_train)

        # instantiate list to store accuracy scores
        accuracy_scores = []

        # iterate over the stages of the model
        for y_pred in ada_discrete.staged_predict(X_test):

            # calculate the accuracy at each stage
            accuracy = 1.0 - (y_pred != y_test).mean()

            # update accuracy scores list
            accuracy_scores.append(accuracy)

        # update the accuracy scores for the entire depth
        accuracy_by_depth.append(accuracy_scores)

    # plot the learning curve for different depths
    plt.figure(figsize=(8, 6))
    max_scores = []
    for i, depth in enumerate(max_depths):
        num_estimators_used = len(accuracy_by_depth[i])
        plt.plot(range(1, num_estimators_used + 1), accuracy_by_depth[i], label = f'Depth {depth}')

        max_score = max(accuracy_by_depth[i]) 
        max_scores.append(max_score)  

   
   # max score
    max_of_max_scores = max(max_scores)
    plt.axhline(max_of_max_scores, 
                color = 'grey', 
                linestyle = '--', 
                linewidth = 1, 
                label = f'Max Accuracy: {max_of_max_scores:.4f}')

        
    # labels
    plt.xlabel('Number of Estimators ~ Trees', weight = 'bold')
    plt.ylabel('Accuracy Score', weight = 'bold')
    plt.title(f'AdaBoost Learning Curve for Different Tree Depths \n Learning Rate = {learning_rate}', 
              weight = 'bold')
    plt.legend()
    
    # save fig
    save_path = (save_path)
    plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
    plt.close()

##########################################################################################################
########################################### main #########################################################
##########################################################################################################

# generate fig, learning rate = .001
generate_adaboost_fig('figs/adaboost_001.png', .001, 2500)

# generate fig, learning rate = .01
generate_adaboost_fig('figs/adaboost_01.png', .01, 2500)