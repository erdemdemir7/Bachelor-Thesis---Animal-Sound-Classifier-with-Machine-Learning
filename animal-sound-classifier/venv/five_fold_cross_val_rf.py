import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from functions import *
from sklearn.metrics import confusion_matrix
from pycm import *
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)

def rf_five_fold(estimator, X, y, output):

    # Define 5-Fold Lists
    scores = []
    mean_squareds = []
    mean_absolutes = []

    #fitting random forest on the dataset
    rfc = RandomForestClassifier(n_estimators = estimator)

    #5-Fold-Cross-Validation
    cv = KFold(n_splits=5, random_state=None, shuffle=True)

    counter = 0 # To state the model

    for train_index, test_index in cv.split(X):
        counter += 1
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        scores.append(rfc.score(X_test, y_test))
        y_pred = rfc.predict(X_test)
        mean_absolutes.append(metrics.mean_absolute_error(y_test, y_pred))
        mean_squareds.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        cm = confusion_matrix(y_test, y_pred)
        tmp = ConfusionMatrix(actual_vector=y_test, predict_vector=y_pred)

        cmd = pd.DataFrame(cm, index=tmp.classes, columns=tmp.classes)
        '''
        # For each Classification
        # Printing metrics
        print(f'\nRFC\nModel {counter}:\n')
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('Accuracy of Model:', rfc.score(X_test, y_test))
        
        # Plotting results
        plt.figure(figsize=(7, 7))
        sns.heatmap(cmd, annot=True, cmap='RdPu')
        txt1 = 'RFC Linear Kernel '
        txt2 = '\nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred))
        txt3 = f' Model {counter}:'
        txt = txt1 + txt3 + txt2
        plt.title(txt)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
        # Printing Confusion matrix details
        print("\n\nlabel precision recall")
        for label in range(len(cm)):
            print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")
    
        print("\nprecision total:", precision_macro_average(cm))
        print("recall total:", recall_macro_average(cm))
    
        print(f'\nAccuracy: {accuracy(cm)}')
        print('\n--------------------------------------------\n\n')
        '''

    # Accuracy, MSE, MAE Results
    tmp = f'\nScores: {scores}'
    output.append(tmp)
    # print(tmp)

    tmp = f'\nRoot Mean Squared Errors: {mean_squareds}'
    output.append(tmp)
    # print(tmp)

    tmp = f'\nMean Absolute Errors: {mean_absolutes}'
    output.append(tmp)
    # print(tmp)

    # Mean of Accuracy, MSE and MAE Results
    disp_mean = "{:.2f}".format(100 * np.mean(scores))
    tmp = f'\nAccuracy of 5-Fold-Cross-Validation \nMean: {disp_mean}%'
    output.append(tmp)
    # print(tmp)

    tmp = f'\nRMSE of 5-Fold-Cross-Validation \nMean: {np.mean(mean_squareds)}'
    output.append(tmp)
    # print(tmp)

    tmp = f'\nMAE of 5-Fold-Cross-Validation \nMean: {np.mean(mean_absolutes)}\n'
    output.append(tmp)
    # print(tmp)

    # Std. Dev. of Accuracy Results
    disp_std = "{:.2f}".format(np.std(scores))
    tmp = f'Standard Deviation of Accuracy: {disp_std}\n'
    output.append(tmp)
    # print(tmp)

    return output

# Additional Codes provided for Result Module
'''
# Plotting results
plt.scatter(y_test, y_pred)
plt.xlabel('True_Values')
plt.ylabel('Predictions')
plt.title(f'Model {counter}:')
plt.show()
'''

'''
# Sklearn cross val.
c_scores = cross_val_score(rfc, X, y, cv=5)
print(f'\nCross-validated scores: {c_scores}')

# Make cross validated predictions
predictions = cross_val_predict(rfc, X, y, cv=5)
plt.scatter(y, predictions)
plt.xlabel('True_Values')
plt.ylabel('Predictions')
plt.title('Model Pred:')
plt.show()
'''