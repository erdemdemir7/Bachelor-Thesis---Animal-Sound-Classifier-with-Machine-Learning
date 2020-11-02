import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from functions import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pycm import *
from five_fold_cross_val_svm import svm_five_fold


# For Input purposes
'''
Works well with linear kernel but not with rbf kernel

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
'''
def SVM_CLSF(preferences):
    # Train-set initialized
    X, y = preprocess()
    is_optimized = preferences[-2]
    is_5_fold = preferences[-1]
    if is_optimized:
        kernel = 'linear'
        C = 1
    else:
        kernel = preferences[1]
        C = int(preferences[0])

    # SVM Classification
    classifier = svm.SVC(kernel= kernel, C= C)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None, shuffle = True)

    #fitting on the entire data
    model = classifier.fit(X_train, y_train)
    #predictions = classifier.predict(X_test)
    predictions = classifier.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    tmp = ConfusionMatrix(actual_vector=y_test, predict_vector=predictions)

    cmd = pd.DataFrame(cm, index=tmp.classes, columns=tmp.classes)

    # Plotting results
    plt.clf()
    plt.figure(figsize=(7,7))
    sns.heatmap(cmd, annot=True, cmap='RdPu')
    plt.title(f'SVM {kernel} Kernel Accuracy: {str(accuracy_score(y_test, predictions))[:4]}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('SVM-CM.png')
    #plt.show()

    # Output is initialized
    output = []
    tmp = ''
    '''
    tmp = "\nlabel precision recall"
    output.append(tmp)
    # print(tmp)

    for label in range(len(cm)):
        tmp = f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}"
        output.append(tmp)
        # print(tmp)

    tmp = f"\nprecision total: {precision_macro_average(cm)}"
    output.append(tmp)
    # print(tmp)

    tmp = f"\nrecall total: {recall_macro_average(cm)}"
    output.append(tmp)
    # print(tmp)

    tmp = f'\nAccuracy: {accuracy(cm)}'
    output.append(tmp)
    # print(tmp)
    '''
    # checking the accuracy of the model
    tmp = f'\nModel SVM:\n'
    output.append(tmp)
    # print(tmp)

    tmp = f'\nMean Absolute Error: {metrics.mean_absolute_error(y_test, predictions)}'
    output.append(tmp)
    # print(tmp)

    tmp = f'\nMean Squared Error: {metrics.mean_squared_error(y_test, predictions)}'
    output.append(tmp)
    # print(tmp)

    tmp = f'\nRoot Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, predictions))}'
    output.append(tmp)
    # print(tmp)

    tmp = f'\nAccuracy of Model: {classifier.score(X_test, y_test)}'
    output.append(tmp)
    # print(tmp)

    if is_5_fold:
        output = svm_five_fold(kernel,C, X, y, output)

    return output


#Scatter plot
'''
plt.scatter(y_test, predictions)
plt.xlabel('True_Values')
plt.ylabel('Predictions')
plt.title('Model SVM:')
plt.show()
'''

'''
# Prepare submission (mapped to file-names)
subm = pd.DataFrame()
subm['song'] = test['song']
subm['bird'] = str_preds
subm.to_csv('submission.csv', index=False)
'''
