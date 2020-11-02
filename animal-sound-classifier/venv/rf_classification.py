import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from functions import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pycm import *
from five_fold_cross_val_rf import rf_five_fold



def RandomForest(preferences):
    # Train-set initialized
    X, y = preprocess()
    is_optimized = preferences[-2]
    is_5_fold = preferences[-1]

    if is_optimized:
        estimator = 150
    else:
        estimator = int(preferences[0])


    #fitting random forest on the dataset
    rfc = RandomForestClassifier(n_estimators = estimator)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None, shuffle = True)

    #fitting on the entire data
    model = rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    tmp = ConfusionMatrix(actual_vector=y_test, predict_vector=predictions)

    cmd = pd.DataFrame(cm, index=tmp.classes, columns=tmp.classes)

    # Plotting results
    plt.clf()
    plt.figure(figsize=(7,7))
    sns.heatmap(cmd, annot=True, cmap='RdPu')
    plt.title('RFC \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, predictions)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('RF-CM.png')
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
    tmp = f'\nModel RF:\n'
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

    tmp = f'\nAccuracy of Model: {rfc.score(X_test, y_test)}'
    output.append(tmp)
    # print(tmp)


    if is_5_fold:
        output = rf_five_fold(estimator, X, y, output)

    return output

'''
# Plotting results
plt.scatter(y_test, predictions)
plt.xlabel('True_Values')
plt.ylabel('Predictions')
plt.title('Model Random_Forest:')
plt.show()
'''


'''
#Model
#print(f'\n\n {model}')

# Prepare submission
str_preds, _ = proba2labels(rfc.predict_proba(test_data.drop('bird', axis = 1).values), i2c, k=3) # For submission.csv
subm = pd.DataFrame()
subm['song'] = test['song']
subm['bird'] = str_preds
subm.to_csv('submission.csv', index=False)

def proba2labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids

'''

