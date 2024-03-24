from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
digits = load_digits()
X = digits.data
y = digits.target

# Train/test split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

# Train and test the model
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

# Print results
print('\nTrain accuracy: {:.2f}'.format(accuracy_score(ytrain, model.predict(Xtrain))))
print('Test accuracy: {:.2f}\n'.format(accuracy_score(ytest, y_model)))


# Plot the confusion matrix
mat = confusion_matrix(ytest, y_model)
sns.heatmap(mat, square=True, annot=True, cbar=False, cmap='OrRd')
plt.title('Gaussian Naive Bayes')
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.savefig("figs/template.png")
#plt.show()
plt.close()
