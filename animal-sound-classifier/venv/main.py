### Main program will run from this module. Connection will be made and the main program will run on cloud.
### After receiving input data and input parameter form from the GUI which was sent via network, classification or clustering will be executed.
### Results will be created on the result-evaluation module, and will be sent back to GUI to be printed and database to be kept.
### This module will be implemented when network and GUI packages of the project is ready.

from dbscan_clustering import *
from k_means_clustering import *
from rf_classification import *
from svm_classification import *
#from GUI import run
from train_test_create import Create


def execute(inputs):
    result = list()
    def check(name, preferences, result):
        if name.upper() == 'RF':
           result= RandomForest(preferences)
        elif name.upper() == 'SVM':
            result= SVM_CLSF(preferences)
        elif name.upper() == 'DBSCAN':
            result= DBSCAN_CLUSTER(preferences)
        elif name.upper() == 'KMEANS':
            result= KMEANS_CLUSTER(preferences)
        return result

    #inputs = get_output()
    print(inputs)
    is_labelled = inputs[0]
    pref = inputs[1]
    lst = list(pref.keys())
    print('Implemented\n')
    # To state whether Classification or Clustering is running
    if is_labelled:
        print('Classification is running\n')
    else:
        print('Clustering is running\n')
    # If only one option selected
    if len(pref.get(lst[1])) == 0:
        name1 = lst[0]
        params1 = pref.get(name1)
        result= check(name1, params1, result)
    elif len(pref.get(lst[0])) == 0:
        name1 = lst[1]
        params1 = pref.get(name1)
        result= check(name1, params1, result)
    else:
        name1 = lst[0]
        name2 = lst[1]
        params1 = pref.get(name1)
        params2 = pref.get(name2)
        result1= check(name1, params1, result)
        result2= check(name2, params2, result)
        for x in result1:
            result.append(x)
        for y in result2:
            result.append(y)
    return result

'''
if __name__ == "__main__":
    Create()
    run()
'''

    # Network

    # Cloud



